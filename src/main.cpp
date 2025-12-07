#pragma once

#include <map>
#include <vector>
#include <iostream>
#include <mutex>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <DemBones/DemBonesExt.h>
#include <DemBones/MatBlocks.h>

#include <maya/MFnIkJoint.h>
#include <maya/MDagModifier.h>
#include <maya/MItGeometry.h>

#include <Python.h>
#include "Py_MString.h"

#include "utils.h"

namespace py = pybind11;
using namespace std;
using namespace Eigen;
using namespace Dem;
using namespace MayaDemBoneUtils;


class DemBonesModel : public DemBonesExt<double, float> {
public:
	using Super = DemBonesExt<double, float>;
	int sF = 1001;
	int eF = 1010;
	vector<string> bonesMaya;
	vector<double> weightsMaya;
	MDagPathArray bonesMayaDagpathArray;
	map<string, MMatrix> bindMatricesMaya;
	map<string, map<int, MMatrix>> animMatricesMaya;
	map<string, MTransformationMatrix::RotationOrder> rotOrderMaya;

	double tolerance;
	int patience;
	char* lockWeightsSet = "demLock";
    char* lockWeightsAttr = "demLock";

    MString sourceDagShapePathName = "";
    MString targetDagShapePathName = "";

	// New: allow specifying the number of bones to auto-create when the scene has no bones.
	// If -1 (default), one bone will be created so computation can proceed.
	int numBones = -1;

	DemBonesModel() : tolerance(1e-3), patience(3) { nIters = 30; clear(); }

	void cbIterBegin() {
		LOG("  iteration #" << iter );
	}

	bool cbIterEnd() {
		double err = rmse();
		LOG("    rmse = " << err );
		if ((err < prevErr * (1 + weightEps)) && ((prevErr - err) < tolerance * prevErr)) {
			np--;
			if (np == 0) {
				LOG("  convergence is reached" );
				return true;
			}
		}
		else np = patience;
		prevErr = err;
		return false;
	}

	void cbInitSplitBegin() {}

	void cbInitSplitEnd() {}

	void cbWeightsBegin() {}

	void cbWeightsEnd() { LOG("    updated weights..." ); }

	void cbTranformationsBegin() {}

	void cbTransformationsEnd() { LOG("    updated transforms..." ); }

	bool cbTransformationsIterEnd() { return false; }

	bool cbWeightsIterEnd() { return false; }

	void extractSource(MDagPath& dag, MFnMesh& mesh) {
		MatrixXd wd(0, 0);
		MIntArray indices;
		MDoubleArray weights;
		MDagPath boneParentMaya;
		//MDagPathArray bonesMaya;
		MDagPathArray& bonesMaya = bonesMayaDagpathArray;
		map<string, MatrixXd, less<string>, aligned_allocator<pair<const string, MatrixXd>>> mT;
		map<string, VectorXd, less<string>, aligned_allocator<pair<const string, VectorXd>>> wT;
		map<string, Matrix4d, less<string>, aligned_allocator<pair<const string, Matrix4d>>> bindMatrices;
		bool hasKeyFrame = false;

		time.setValue(sF);
		anim.setCurrentTime(time);

		// update model: bind vertex positions
		status = mesh.getPoints(points, MSpace::kWorld);
		CHECK_MSTATUS_AND_THROW(status);

		u.resize(3, nV);

		#pragma omp parallel for
		for (int i = 0; i < nV; i++) {
			MPoint point = points[i];
			u.col(i) << point.x, point.y, point.z;
		}

		// update model: face connections
		int nFV = mesh.numPolygons();
		fv.resize(nFV);

		for (int i = 0; i < nFV; i++) {
			mesh.getPolygonVertices(i, indices);
			int length = indices.length();
			for (int j = 0; j < length; j++)
				fv[i].push_back((int)indices[j]);
		}

		// update model: weights and skeleton
		MItDependencyGraph graphIter(dag.node(), MFn::kSkinClusterFilter, MItDependencyGraph::kUpstream);
		MObject rootNode = graphIter.currentItem(&status);

		if (MS::kSuccess == status) {
			// query bones
			MFnSkinCluster skinCluster(rootNode);
			nB = skinCluster.influenceObjects(bonesMaya, &status);
			CHECK_MSTATUS_AND_THROW(status);

			// get bones names
			boneName.resize(nB);
			for (int j = 0; j < nB; j++) {
				string name = bonesMaya[j].partialPathName().asUTF8();
				boneName[j] = name;
				boneIndex[name] = j;
			}

			// get bone weights
			for (int j = 0; j < nB; j++) {
				status = skinCluster.getWeights(dag, MObject::kNullObj, j, weights);
				CHECK_MSTATUS_AND_THROW(status);

				wT[boneName[j]] = VectorXd::Zero(nV);
				for (int k = 0; k < nV; k++)
					wT[boneName[j]](k) = weights[k];
			}
		}

		parent.resize(nB);
		bind.resize(nS * 4, nB * 4);
		preMulInv.resize(nS * 4, nB * 4);
		rotOrder.resize(nS * 3, nB);
		orient.resize(nS * 3, nB);
		lockM.resize(nB);
		lockW = VectorXd::Zero(nV);

		// update model: weights
		if (wT.size() != 0) {
			wd = MatrixXd::Zero(nB, nV);
			for (int j = 0; j < nB; j++)
				wd.row(j) = wT[boneName[j]].transpose();
		}

		// update model: weights locked
		MString colourSet = lockWeightsSet;
		if (mesh.hasColorChannels(colourSet)) {
			MColorArray colours;
			status = mesh.getVertexColors(colours, &colourSet);
			CHECK_MSTATUS_AND_THROW(status);

			for (int c = 0; c < (int)colours.length(); c++)
				lockW(c) = Conversion::toGrayScale(colours[c]);
		}

		// update model: skeleton
		for (int j = 0; j < nB; j++) {
			// get name
			string name = boneName[j];

			// get parent
			MObject boneObj = bonesMaya[j].node();
			MFnDagNode boneDagFn(boneObj, &status);
			CHECK_MSTATUS_AND_THROW(status);
			MObject boneParentObj = boneDagFn.parent(0);

			if (!boneParentObj.isNull() && boneParentObj.hasFn(MFn::kJoint)) {
				status = MDagPath::getAPathTo(boneParentObj, boneParentMaya);
				CHECK_MSTATUS_AND_THROW(status);

				string parentName = boneParentMaya.partialPathName().asUTF8();
				if (boneIndex.find(parentName) == boneIndex.end())
					parent(j) = -1;
				else
					parent(j) = boneIndex[parentName];
			}
			else {
				parent(j) = -1;
			}

			// get bind matrix
			mT[name].resize(nF * 4, 4);
			Matrix4d bindMatrix = Conversion::toMatrix4D(bonesMaya[j].inclusiveMatrix());
			bind.blk4(0, j) = bindMatrix;
			bindMatrices[name] = bindMatrix;

			// get rotation order
			MPlug rotateOrderPlug = boneDagFn.findPlug("rotateOrder", &status);
			//CHECK_MSTATUS_AND_THROW(status);
			int rotateOrder = 0;
			if (status.statusCode() == MStatus::kSuccess)
			{
				rotateOrder = rotateOrderPlug.asInt();
			}

			switch (rotateOrder) {
				case 0: {
					rotOrder.vec3(0, j) = Vector3i(0, 1, 2);
					rotOrderMaya[name] = MTransformationMatrix::RotationOrder::kXYZ;
					break;
				}
				case 1: {
					rotOrder.vec3(0, j) = Vector3i(1, 2, 0);
					rotOrderMaya[name] = MTransformationMatrix::RotationOrder::kYZX;
					break;
				}
				case 2: {
					rotOrder.vec3(0, j) = Vector3i(2, 0, 1);
					rotOrderMaya[name] = MTransformationMatrix::RotationOrder::kZXY;
					break;
				}
				case 3: {
					rotOrder.vec3(0, j) = Vector3i(0, 2, 1);
					rotOrderMaya[name] = MTransformationMatrix::RotationOrder::kXZY;
					break;
				}
				case 4: {
					rotOrder.vec3(0, j) = Vector3i(1, 0, 2);
					rotOrderMaya[name] = MTransformationMatrix::RotationOrder::kYXZ;
					break;
				}
				case 5: {
					rotOrder.vec3(0, j) = Vector3i(2, 1, 0);
					rotOrderMaya[name] = MTransformationMatrix::RotationOrder::kZYX;
					break;
				}
			}

			// get joint orient
			MPlug jointOrientPlug = boneDagFn.findPlug("jointOrient", &status);
			//CHECK_MSTATUS_AND_THROW(status);
			double jointOrientX = 0.0;
			double jointOrientY = 0.0;
			double jointOrientZ = 0.0;
			if (status.statusCode() == MStatus::kSuccess)
			{
				jointOrientX = jointOrientPlug.child(0).asMAngle().asDegrees();
				jointOrientY = jointOrientPlug.child(1).asMAngle().asDegrees();
				jointOrientZ = jointOrientPlug.child(2).asMAngle().asDegrees();
			}

			orient.vec3(0, j) = Vector3d(jointOrientX, jointOrientY, jointOrientZ);

			// get pre multiply inverse
			if (!boneParentObj.isNull() && parent(j) == -1) {
				status = MDagPath::getAPathTo(boneParentObj, boneParentMaya);
				CHECK_MSTATUS_AND_THROW(status);

				Matrix4d gp = Conversion::toMatrix4D(boneParentMaya.inclusiveMatrix());
				preMulInv.blk4(0, j) = gp.inverse();
			}
			else {
				preMulInv.blk4(0, j) = Matrix4d::Identity();
			}

			// get dem lock
			MPlug demLockPlug = boneDagFn.findPlug(lockWeightsAttr, &status);
			if (MS::kSuccess == status) 
				lockM(j) = demLockPlug.asInt(); 
			else lockM(j) = 0;
		}

		// update model: skeleton animation
		m.resize(nF * 4, nB * 4);
		for (int k = sF; k < eF + 1; k++) {
			time.setValue(k);
			anim.setCurrentTime(time);
			int num = k - sF;

			for (int j = 0; j < nB; j++) {
				// get name
				string name = boneName[j];

				// set matrix
				Matrix4d matrix = Conversion::toMatrix4D(bonesMaya[j].inclusiveMatrix());
				mT[name].blk4(num, 0) = matrix * bindMatrices[name].inverse();
			}
		}
		for (int j = 0; j < nB; j++) {
			m.block(0, j * 4, nF * 4, 4) = mT[boneName[j]];
		}

		// update model: animation state
		MString transformAttributes[9] = { "tx", "ty", "tz", "rx", "ry", "rz", "sx", "sy", "sz" };
		for (int j = 0; j < nB; j++) {
			string nj = boneName[j];

			for (int k = 0; k < 9; k++) {
				MFnDependencyNode node(bonesMaya[j].node(), &status);
				CHECK_MSTATUS_AND_THROW(status);

				MPlug plug = node.findPlug(transformAttributes[k], &status);
				CHECK_MSTATUS_AND_THROW(status);

				if (plug.isDestination()) {
					hasKeyFrame = true;
					break;
				}
			}
		}

		w = (wd / nS).sparseView(1, 1e-20);
		lockW = lockW /= (double)nS;
		if (!hasKeyFrame) m.resize(0, 0);

		// report
		LOG("extracted source" );
		LOG("  " << nV << " vertices" );
		if (nB != 0) LOG("  " << nB << " joints" );
		if (hasKeyFrame) LOG("  keyframes found" );
		if (w.size() != 0) LOG("  skinning found" );
	}

	void extractTarget(MFnMesh& mesh) {
		// update model: animated vertex positions
		for (int k = sF; k < eF + 1; k++) {
			time.setValue(k);
			anim.setCurrentTime(time);

			int num = k - sF;
			fTime(num) = time.asUnits(MTime::kSeconds);
			status = mesh.getPoints(points, MSpace::kWorld);
			CHECK_MSTATUS_AND_THROW(status);

			#pragma omp parallel for
			for (int i = 0; i < nV; i++) {
				MPoint point = points[i];
				v.col(i).segment<3>(num * 3) << (float)point.x, (float)point.y, (float)point.z;
			}
		}

		// update model: subject ids
		subjectID.resize(nF);
		for (int s = 0; s < nS; s++)
			for (int k = fStart(s); k < fStart(s + 1); k++)
				subjectID(k) = s;

		LOG("extracted target" );
	}

	void compute(string& source, string& target, int& startFrame, int& endFrame) {
		// log parameters
		LOG("parameters" );
		LOG("  source                   = " << source );
		LOG("  target                   = " << target );
		LOG("  start_frame              = " << startFrame );
		LOG("  end_frame                = " << endFrame );
		LOG("  num_iterations           = " << nIters );
		LOG("  patience                 = " << patience );
		LOG("  tolerance                = " << tolerance );
		LOG("  num_transform_iterations = " << nTransIters );
		LOG("  num_weight_iterations    = " << nWeightsIters );
		LOG("  translation_affine       = " << transAffine );
		LOG("  translation_affine_norm  = " << transAffineNorm );
		LOG("  max_influences           = " << nnz );
		LOG("  weights_smooth           = " << weightsSmooth );
		LOG("  weights_smooth_step      = " << weightsSmoothStep );
		LOG("  weights_epsilon          = " << weightEps );

		// variables
		prevErr = -1;
		np = patience;

		// get geometry
		MDagPath sourcePath = Conversion::toMDagPath(source, true);
		MDagPath targetPath = Conversion::toMDagPath(target, true);

		MFnMesh sourceMeshFn(sourcePath, &status);
		CHECK_MSTATUS_AND_THROW(status);
		MFnMesh targetMeshFn(targetPath, &status);
		CHECK_MSTATUS_AND_THROW(status);

        sourceDagShapePathName = sourcePath.fullPathName();
        targetDagShapePathName = targetPath.fullPathName();

		// update model
		nS = 1;
		sF = startFrame;
		eF = endFrame;
		nF = endFrame - startFrame + 1;
		nV = sourceMeshFn.numVertices();
		int cF = (int)anim.currentTime().value();

		if (sF >= eF)
			Conversion::RaiseException("Start frame is not allowed to be equal or larger than the end frame.");
		if (nV != sourceMeshFn.numVertices())
			Conversion::RaiseException("Vertex count between source and target do not match.");

		v.resize(3 * nF, nV);
		fTime.resize(nF);
		fStart.resize(nS + 1);
		fStart(0) = 0;
		fStart(1) = nF;

		// update model: source + target
		extractTarget(targetMeshFn);
		extractSource(sourcePath, sourceMeshFn);

		// initialize model
		InitBones();

		// compute model
		LOG("computing" );
		DemBonesExt<double, float>::compute();

		// compute transformations + weights
		VectorXd tVal, rVal;
		MatrixXd lr, lt, gb, lbr, lbt;
		computeRTB(0, lr, lt, gb, lbr, lbt, false);


		for (int j = 0; j < nB; j++) {
			std::lock_guard<std::mutex> lock(mtx_);
			string name = boneName[j];
			bonesMaya.push_back(name);
			MVector translate = MVector(lbt(0, j), lbt(1, j), lbt(2, j));
			MVector rotate = MVector(lbr(0, j), lbr(1, j), lbr(2, j));
			
			bindMatricesMaya[name] = Conversion::toMMatrix(translate, rotate, rotOrderMaya[name]);

			tVal = lt.col(j);
			rVal = lr.col(j);

			for (int k = sF; k < eF + 1; k++) {
				int num = k - sF;
				MVector translate = MVector(tVal(num * 3), tVal((num * 3) + 1), tVal((num * 3) + 2));
				MVector rotate = MVector(rVal(num * 3), rVal((num * 3) + 1), rVal((num * 3) + 2));
				animMatricesMaya[name][k] = Conversion::toMMatrix(translate, rotate, rotOrderMaya[name]);
			}
		}

		LOG("Converting to Maya Weights" );
		weightsMaya.resize(nB * nV);
		Eigen::SparseMatrix<double> wT = w.transpose();
		for (int j = 0; j < nB; j++)
		{
			std::lock_guard<std::mutex> lock(mtx_);
			for (Eigen::SparseMatrix<double>::InnerIterator it(wT, j); it; ++it)
				weightsMaya[((int)it.row() * nB) + j] = it.value();
		}


		time.setValue(cF);
		anim.setCurrentTime(time);
	}

	array<double, 16> bindMatrix(string& bone) {
		if (bindMatricesMaya.find(bone) == bindMatricesMaya.end())
			Conversion::RaiseException("Provided influence is not valid.");

		return Conversion::toMatrixArray(bindMatricesMaya[bone]);
	}

	array<double, 16> animMatrix(string& bone, int& frame) {
		if (animMatricesMaya.find(bone) == animMatricesMaya.end())
			Conversion::RaiseException("Provided bone is not valid.");
		if (animMatricesMaya[bone].find(frame) == animMatricesMaya[bone].end())
			Conversion::RaiseException("Provided frame is not valid.");

		return Conversion::toMatrixArray(animMatricesMaya[bone][frame]);
	}

	void clear() {
		DemBonesExt<double, float>::clear();
		points.clear();
		boneIndex.clear();
	}

	// initialize model
	void InitBones() 
	{

		if (nB > 0) return;

		// If there are truly no influences (no bones) in the scene, call init() to generate clusters / initial bind,
		// then create joints in Maya so joint positions are based on centers of bind matrices produced by init() (bind.blk4).
		int createB = (numBones > 0) ? numBones : 1;
		LOG("No influences found. Initializing bones (compute clusters): target = " << createB );

		// Set target number of bones and call init(). init() performs initialization (LBG-VQ etc.) producing label/bind/w etc.
		nB = createB;
		if (nInitIters <= 0) nInitIters = 10;

		LOG("Init Joints" );
		init(); // now bind / label / w etc. should be initialized

		// Place joints according to bind matrices generated by init()
		MStatus& st = status;
		bonesMayaDagpathArray.clear();
		LOG("Start Build Bind Matrix" );

		// Create joints and record dagPath
		for (int j = 0; j < nB; ++j) {
            std::lock_guard<std::mutex> lock(mtx_); // Always lock when creating Maya objects.
			// Try to read translation from bind matrices (bind is a DemBonesExt member)
			Matrix4d bindMat = Matrix4d::Identity();
			// If bind data exists and is valid, prefer reading position from bind.blk4
			if ((int)bind.rows() >= 4 && (int)bind.cols() >= (j+1)*4) {
				bindMat = bind.blk4(0, j);
			} else {
				// Fallback to point cloud centroid
				bindMat(0, 3) = u.row(0).mean();
				bindMat(1, 3) = u.row(1).mean();
				bindMat(2, 3) = u.row(2).mean();
			}

			MVector pos((double)bindMat(0, 3), (double)bindMat(1, 3), (double)bindMat(2, 3));
			MFnIkJoint jointFn;
			MObject jObj;
			jObj = jointFn.create(); // Workaround with Maya API bug.
            
			if (jObj.isNull())
			{
				Conversion::RaiseException("Failed to create joint in Maya.");
				break;
			}
			// jointFn.setObject(jObj); // Ensure update.

			string name = Conversion::FormatString("def_joint_%d", j);
			MString jname(name.c_str());
			jname = jointFn.setName(jname, false, &st);
			CHECK_MSTATUS_AND_THROW(st);

			// Place joint at translation position from bind matrix (use world space so reparenting won't change world position)
			st = jointFn.setTranslation(pos, MSpace::kTransform);
			CHECK_MSTATUS_AND_THROW(st);

			// Record bone names and indices to the model
			boneName.push_back(name);
			boneIndex[name] = j;
			bonesMayaDagpathArray.append(jointFn.dagPath());
		}

		// If parent information exists, build parent-child relationships according to parent array
		// parent is ArrayXi, -1 denotes root
		// Use MDagModifier to batch reparent and call doIt() once at the end
		MDagModifier dagMod;
		bool hasParentInfo = (parent.size() == nB);
		if (hasParentInfo) {
			std::lock_guard<std::mutex> lock(mtx_);
			for (int j = 0; j < nB; ++j) {
				int p = parent(j);
				if (p != -1 && p >= 0 && p < nB) {
					MObject childObj = bonesMayaDagpathArray[j].node();
					MObject parentObj = bonesMayaDagpathArray[p].node();
					st = dagMod.reparentNode(childObj, parentObj);
					CHECK_MSTATUS_AND_THROW(st);
				}
			}
			st = dagMod.doIt();
			CHECK_MSTATUS_AND_THROW(st);
		}
	}

	void updateResultSkinWeight(string& skin_mesh)
	{
		std::lock_guard<std::mutex> lock(mtx_);

		MDagPath mayaSkinMeshDagpath = Conversion::toMDagPath(skin_mesh, true);
		MObject skinMeshObj = mayaSkinMeshDagpath.node(&status);
		CHECK_MSTATUS_AND_THROW(status);

		MItDependencyGraph itDG(skinMeshObj,
			MFn::kSkinClusterFilter,
			MItDependencyGraph::kUpstream,
			MItDependencyGraph::kDepthFirst,
			MItDependencyGraph::kNodeLevel,
			&status);

		bool bFoundValidSkinCluster = false;
		std::shared_ptr<MFnSkinCluster> fnMayaSkinMesh;
		for (; !itDG.isDone(); itDG.next()) {
			MObject skinObj = itDG.currentItem();
			if (skinObj.hasFn(MFn::kSkinClusterFilter)) {
				fnMayaSkinMesh = std::make_shared<MFnSkinCluster>(skinObj, &status);
				CHECK_MSTATUS_AND_THROW(status);
				bFoundValidSkinCluster = true;
				break;
			}
		}

		if (!bFoundValidSkinCluster)
		{
			Conversion::RaiseException("No valid skin cluster found on the provided mesh.");
            return;
		}

		//if (!bFoundValidSkinCluster)
		//{
		//	// NOTE: This part has been move to Python script.
		//	/*
		//		C++ Part looks really hacky.
		//		We create skin cluster here if not found,
		//	*/
		//	LOG("Creat SkinCluster for Joints" );
		//	MString cmd = "skinCluster -tsb";
		//	for (int j = 0; j < nB; j++) {
		//		string name = boneName[j];

		//		MMatrix resetPostMatrix = bindMatricesMaya[name];
		//		MDagPath bonePath = Conversion::toMDagPath(name, false);
		//		MFnTransform boneDagFn(bonePath, &status);
		//		boneDagFn.set(resetPostMatrix);

		//		MString jname(name.c_str());
		//		cmd += " " + jname;
		//	}

		//	cmd += " " + mayaSkinMeshDagpath.partialPathName();
		//	cmd += ";";

		//	LOG(Conversion::FormatString("Exec Command: %s", cmd.asChar()) <<endl );
		//	MGlobal::executeCommandStringResult(cmd, false, false, &status);
		//	if (status.statusCode() != MStatus::kSuccess)
		//	{
		//		MGlobal::displayWarning("Create skin cluster -> " + status.errorString());
		//	}

		//	// Re-fetch skin cluster
		//	MItDependencyGraph graphIter(skinMeshObj, MFn::kSkinClusterFilter, MItDependencyGraph::kUpstream);
		//	for (; !graphIter.isDone(); graphIter.next()) {
		//		MObject skinObj = graphIter.currentItem();
		//		if (skinObj.hasFn(MFn::kSkinClusterFilter)) {
		//			fnMayaSkinMesh = std::make_shared<MFnSkinCluster>(skinObj, &status);
		//			CHECK_MSTATUS_AND_THROW(status);
		//			bFoundValidSkinCluster = true;
		//			break;
		//		}
		//	}
		//}

		if (bFoundValidSkinCluster)
		{
			MIntArray indices;

			indices.setLength(bonesMaya.size());
			for (auto i = 0; i < bonesMaya.size(); ++i) {
				indices[i] = i;
			}

			LOG("Set weight to mesh" );
			status = fnMayaSkinMesh->setWeights(
				mayaSkinMeshDagpath,
				fnMayaSkinMesh->getComponentAtIndex(0),
				indices,
				Conversion::toMDoubleArray(weightsMaya)
			);
			CHECK_MSTATUS_AND_THROW(status);
		}

	}

private:
	double prevErr;
	int np;

	MStatus status;
	MTime time;
	MAnimControl anim;
	MPointArray points;
	map<string, int> boneIndex;

	// Add Lock
	mutable std::mutex mtx_;
} model;


PYBIND11_MODULE(_core, m) {
	py::class_<DemBonesModel>(m, "DemBones")
		.def(py::init<>())
		.def_readwrite("bind_update", &DemBonesModel::bindUpdate, "Bind transformation update, 0=keep original, 1=set translations to p-norm centroids (using #transAffineNorm) and rotations to identity, 2=do 1 and group joints, default = 0")
		.def_readwrite("init_iterations", &DemBonesModel::nInitIters, "Number of initialization iterations (k-means), default = 10")
		.def_readwrite("num_iterations", &DemBonesModel::nIters, "Number of global iterations, default = 30")
		.def_readwrite("num_transform_iterations", &DemBonesModel::nTransIters, "Number of bone transformations update iterations per global iteration, default = 5")
		.def_readwrite("translation_affine", &DemBonesModel::transAffine, "Translations affinity soft constraint, default = 10.0")
		.def_readwrite("translation_affine_norm", &DemBonesModel::transAffineNorm, "p-norm for bone translations affinity soft constraint, default = 4.0")
		.def_readwrite("num_weight_iterations", &DemBonesModel::nWeightsIters, "Number of weights update iterations per global iteration, default = 3")
		.def_readwrite("max_influences", &DemBonesModel::nnz, "Number of non-zero weights per vertex, default = 8")
		.def_readwrite("weights_smooth", &DemBonesModel::weightsSmooth, "Weights smoothness soft constraint, default = 1e-4")
		.def_readwrite("weights_smooth_step", &DemBonesModel::weightsSmoothStep, "Step size for the weights smoothness soft constraint, default = 1.0")
		.def_readwrite("weights_epsilon", &DemBonesModel::weightEps, "Epsilon for weights solver, default = 1e-15")
		.def_readwrite("tolerance", &DemBonesModel::tolerance, "Convergence tolerance, default = 1e-3")
		.def_readwrite("patience", &DemBonesModel::patience, "Number of iterations to wait before declaring convergence, default = 3")
		.def_readwrite("lock_weights_set", &DemBonesModel::lockWeightsSet, "Name of the color set used to lock weights, default = 'demLock'")
		.def_readwrite("num_bones", &DemBonesModel::numBones, "If >0 and no influences found, number of bones to create automatically (default: -1 -> create 1 bone)")
		.def_readonly("start_frame", &DemBonesModel::sF, "Start frame of solver")
		.def_readonly("end_frame", &DemBonesModel::eF, "End frame of solver")
		.def_readonly("influences", &DemBonesModel::bonesMaya, "List of all influences")
		.def_readonly("weights", &DemBonesModel::weightsMaya, "List of weights for all influences and vertices")
		.def("rmse", &DemBonesModel::rmse, "Root mean squared reconstruction error")
		.def("compute", &DemBonesModel::compute, "Skinning decomposition of alternative updating weights and bone transformations", py::arg("source"), py::arg("target"), py::arg("start_frame"), py::arg("end_frame"))
		.def("bind_matrix", &DemBonesModel::bindMatrix, "Get the bind matrix for the provided influence", py::arg("influence"))
		.def("anim_matrix", &DemBonesModel::animMatrix, "Get the animation matrix for the provided influence at the provided frame", py::arg("influence"), py::arg("frame"))
		.def("update_result_skin_weight", &DemBonesModel::updateResultSkinWeight, py::arg("skin_mesh"))
		.def_readonly("skin_mesh_shape_name", &DemBonesModel::sourceDagShapePathName,"sourceDagShapePathName")
		.def_readonly("vertex_anim_mesh_shape_name", &DemBonesModel::sourceDagShapePathName, "sourceDagShapePathName")
		;
}