#pragma once

#include <map>
#include <vector>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <DemBones/DemBonesExt.h>
#include <DemBones/MatBlocks.h>
#include <Python.h>

#include <maya/MFnSkinCluster.h>
#include <maya/MFnAnimCurve.h>
#include <maya/MPlug.h>
#include <maya/MGlobal.h>
#include <maya/MItDependencyGraph.h>
#include <maya/MTransformationMatrix.h>
#include <maya/MAnimControl.h>
#include <maya/MTime.h>
#include <maya/MIntArray.h>
#include <maya/MDoubleArray.h>
#include <maya/MFnIkJoint.h>
#include <maya/MDagModifier.h>

#include "Py_MString.h"
#include "utils.h"


namespace py = pybind11;
using namespace std;
using namespace Eigen;
using namespace Dem;


class DemBonesModel : public DemBonesExt<double, float> {
public:
	using Super = DemBonesExt<double, float>;
	int sF = 0;
	int eF = 100;
	int nBoneCount = 0;

	MDagPathArray bonesMaya;

	std::vector<std::string> boneName;
	std::vector<double> weightsMaya;

	map<string, MMatrix> bindMatricesMaya;
	map<string, map<int, MMatrix>> animMatricesMaya;
	map<string, MTransformationMatrix::RotationOrder> rotOrderMaya;

	double tolerance;
	int patience;
	char * lockWeightsSet = "demLock";
    char* lockBoneAttrName = "demLockBones";

	DemBonesModel() : tolerance(1e-3), patience(3) { nIters = 30; clear(); }



	void cbIterBegin() {
		LOG("  iteration #" << iter << endl);
	}

	bool cbIterEnd() {
		double err = rmse();
		LOG("    rmse = " << err << endl);
		if ((err<prevErr*(1 + weightEps)) && ((prevErr - err)<tolerance*prevErr)) {
			np--;
			if (np == 0) {
				LOG("  convergence is reached" << endl);
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

	void cbWeightsEnd() { LOG("    updated weights..." << endl);}

	void cbTranformationsBegin() {}

	void cbTransformationsEnd() { LOG("    updated transforms..." << endl); }

	bool cbTransformationsIterEnd() { return false; }

	bool cbWeightsIterEnd() { return false; }

	void extractSource(MDagPath& dag, MFnMesh& mesh) {
		MatrixXd wd(0, 0);
		MIntArray indices;
		MDoubleArray weights;
		MDagPath boneParentMaya;
		//MDagPathArray bonesMaya;
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
			CHECK_MSTATUS_AND_THROW(status);

			int rotateOrder = rotateOrderPlug.asInt();
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
			CHECK_MSTATUS_AND_THROW(status);

			double jointOrientX = jointOrientPlug.child(0).asMAngle().asDegrees();
			double jointOrientY = jointOrientPlug.child(1).asMAngle().asDegrees();
			double jointOrientZ = jointOrientPlug.child(2).asMAngle().asDegrees();
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
			MPlug demLockPlug = boneDagFn.findPlug(MString(lockBoneAttrName), &status);
			if (MS::kSuccess == status) lockM(j) = demLockPlug.asInt(); else lockM(j) = 0;
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
		LOG("[INFO] Extracted source" << endl);
		LOG("  " << nV << " vertices" << endl);
		if (nB != 0) LOG("  " << nB << " joints" << endl);
		if (hasKeyFrame) LOG("  keyframes found" << endl);
		if (w.size() != 0) LOG("  skinning found" << endl);
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

		LOG("[INFO] Extracted target" << endl);
	}

	void compute(string& source, string& target, int& startFrame, int& endFrame) {
		// log parameters
		LOG("\n[INFO] Start Compute" << endl);
		LOG("parameters" << endl);
		LOG("  source                   = " << source << endl);
		LOG("  target                   = " << target << endl);
		LOG("  start_frame              = " << startFrame << endl);
		LOG("  end_frame                = " << endFrame << endl);
		LOG("  num_iterations           = " << nIters << endl);
		LOG("  patience                 = " << patience << endl);
		LOG("  tolerance                = " << tolerance << endl);
		LOG("  num_transform_iterations = " << nTransIters << endl);
		LOG("  num_weight_iterations    = " << nWeightsIters << endl);
		LOG("  translation_affine       = " << transAffine << endl);
		LOG("  translation_affine_norm  = " << transAffineNorm << endl);
		LOG("  max_influences           = " << nnz << endl);
		LOG("  weights_smooth           = " << weightsSmooth << endl);
		LOG("  weights_smooth_step      = " << weightsSmoothStep << endl);
		LOG("  weights_epsilon          = " << weightEps << endl);
		
		// variables
		prevErr = -1;
		np = patience;

		// get geometry
		LOG("Convert source dagpath from " << source << endl);
		MDagPath sourcePath = Conversion::toMDagPath(source, true);
		LOG("Convert target dagpath from " << target << endl);
		MDagPath targetPath = Conversion::toMDagPath(target, true);

		MFnMesh sourceMeshFn(sourcePath, &status);
		CHECK_MSTATUS_AND_THROW(status);

		MFnMesh targetMeshFn(targetPath, &status);
		CHECK_MSTATUS_AND_THROW(status);

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
		LOG("[INFO] Extract Target" << endl);
		extractTarget(targetMeshFn);
		LOG("[INFO] Extract Source" << endl);
		extractSource(sourcePath, sourceMeshFn);

		// initialize model
		bool needCreateJoints = (model.boneName.size() == 0);
		double radius;

		if (nB == 0)
		{
			LOG("[INFO] Not skin mesh");
			if(nBoneCount <= 0)
				Conversion::RaiseException("No joint found in. Need to set the number of bones");
			else {
				nB = nBoneCount;
                boneName.resize(nB);

				

				if (needCreateJoints) {
					model.boneName.resize(model.nB);
					for (int j = 0; j < model.nB; j++) {
						ostringstream s;
						s << "joint" << j;
						model.boneName[j] = s.str();
					}
					radius = sqrt((model.u - (model.u.rowwise().sum() / model.nV).replicate(1, model.nV)).cwiseAbs().rowwise().maxCoeff().squaredNorm() / model.nS);
				}
			}
		}

		// compute model
		LOG("computing" << endl);
		DemBonesExt<double, float>::compute();

		// compute transformations + weights
		VectorXd tVal, rVal;
		MatrixXd lr, lt, gb, lbr, lbt;
		computeRTB(0, lr, lt, gb, lbr, lbt, false);

		if (needCreateJoints) createJoints(model.boneName, model.parent, radius);

		for (int j = 0; j < nB; j++) {
			const std::string name = boneName[j];  // boneName
			MVector translate(lbt(0, j), lbt(1, j), lbt(2, j));
			MVector rotate(lbr(0, j), lbr(1, j), lbr(2, j));
			bindMatricesMaya[name] = Conversion::toMMatrix(translate, rotate, rotOrderMaya[name]);


			tVal = lt.col(j);
			rVal = lr.col(j);

			for (int k = sF; k <= eF; ++k) {
				int num = k - sF;
				MVector translate(tVal(num * 3), tVal(num * 3 + 1), tVal(num * 3 + 2));
				MVector rotate(rVal(num * 3), rVal(num * 3 + 1), rVal(num * 3 + 2));
				animMatricesMaya[name][k] = Conversion::toMMatrix(translate, rotate, rotOrderMaya[name]);
			}
		}

		weightsMaya.resize(nB * nV);
		Eigen::SparseMatrix<double> wT = w.transpose();
		for (int j = 0; j < nB; j++)
			for (Eigen::SparseMatrix<double>::InnerIterator it(wT, j); it; ++it)
				weightsMaya[((int)it.row() * nB) + j] = it.value();

		time.setValue(cF);
		anim.setCurrentTime(time);
	}

	void updateResultSkinWeight(string& skin_mesh)
	{
		MDagPath mayaSkinMeshDagpath = Conversion::toMDagPath(skin_mesh, true);
		MObject obj = mayaSkinMeshDagpath.node();

		MItDependencyGraph itDG(obj,
			MFn::kSkinClusterFilter,
			MItDependencyGraph::kUpstream,
			MItDependencyGraph::kDepthFirst,
			MItDependencyGraph::kNodeLevel,
			&status);

		std::shared_ptr<MFnSkinCluster> fnMayaSkinMesh;
		for (; !itDG.isDone(); itDG.next()) {
			MObject skinObj = itDG.currentItem();
			if (skinObj.hasFn(MFn::kSkinClusterFilter)) {
				fnMayaSkinMesh =  std::make_shared<MFnSkinCluster>(skinObj, &status);
				CHECK_MSTATUS_AND_THROW(status);
				break;
			}
		}
		if (!fnMayaSkinMesh)
            Conversion::RaiseException("No skin cluster found for the provided mesh.");

		MIntArray indices;
		unsigned int boneCount = bonesMaya.length();
		indices.setLength(boneCount);

		for (auto i = 0; i < boneCount; ++i) {
			indices[i] = i;
		}

		status = fnMayaSkinMesh->setWeights(
			mayaSkinMeshDagpath,
			MObject::kNullObj,
			indices,
			Conversion::toMDoubleArray(weightsMaya)
		);
		CHECK_MSTATUS_AND_THROW(status);

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

	void applyAnimationAndWeights(std::string& skinMeshName, bool bUpdateJointWeight = false);

	void createJoints(
		const std::vector<std::string>& names,
		const ArrayXi& parent,
		double radius)
	{
		std::vector<MObject> joints(names.size());
		MDagModifier dagMod;

		for (size_t j = 0; j < names.size(); ++j)
		{
			LOG("[INFO] Create new bone <" << names[j] << ">" << endl);
			MObject jointObj = dagMod.createNode("joint", MObject::kNullObj, &status);
			CHECK_MSTATUS_AND_THROW(status);
			MFnIkJoint fnJoint(jointObj, &status);
			CHECK_MSTATUS_AND_THROW(status);

			// Set position of the root joint (e.g., at the origin)
			MVector rootPos(0.0, 0.0, 0.0);
			status = fnJoint.setTranslation(rootPos, MSpace::kWorld);
			CHECK_MSTATUS_AND_THROW(status);

			fnJoint.setName(MString(names[j].c_str()));
			MPlug radiusPlug = fnJoint.findPlug("radius", true, &status);
			if (status.statusCode() == MStatus::kSuccess) {
				radiusPlug.setDouble(radius);
			};

			fnJoint.setRotationOrder(MTransformationMatrix::kXYZ, true);

			joints[j] = jointObj;
		}
		status = dagMod.doIt();
		CHECK_MSTATUS_AND_THROW(status);

		for (size_t j = 0; j < names.size(); ++j)
		{
			int p = parent[j];
			if (p >= 0 && p < (int)names.size())
			{
				dagMod.reparentNode(joints[j], joints[p]);
			}
		}
		status = dagMod.doIt();
		CHECK_MSTATUS_AND_THROW(status);
	}

private:
	double prevErr;
	int np;
	
	MStatus status;
	MTime time;
	MAnimControl anim;
	MPointArray points;
	map<string, int> boneIndex;
} model;


void DemBonesModel::applyAnimationAndWeights(std::string& skinMeshName, bool bUpdateJointWeight)
{

	MDagPath skinMeshDag = Conversion::toMDagPath(skinMeshName, true);
	MObject skinMeshObj = skinMeshDag.node();
    // Apply animation to bones
	const char* transformAttrs[] = { "translateX","translateY","translateZ",
						   "rotateX","rotateY","rotateZ" };

	constexpr int attrsCount = sizeof(transformAttrs) / sizeof(transformAttrs[0]);
	const MTime::Unit currentUnit = MTime::uiUnit();

	for (int j = 0; j < nB; ++j) {
		std::string name = boneName[j];
		MObject boneObj = bonesMaya[j].node();
		MFnDependencyNode boneNode(boneObj);

		for (int attrIndex = 0; attrIndex < attrsCount; ++attrIndex) {
			MPlug plug = boneNode.findPlug(transformAttrs[attrIndex], true);
			LOG(Conversion::FormatString("Processing {}.{}", name, transformAttrs[attrIndex]) << endl);
			MObject animCurveObj;

			MPlugArray connections;
			plug.connectedTo(connections, true, false);
			if (connections.length() > 0 && connections[0].node().hasFn(MFn::kAnimCurve)) {
				animCurveObj = connections[0].node();
			}
			else {
				MFnAnimCurve animFn;
				animCurveObj = animFn.create(plug, MFnAnimCurve::kAnimCurveTL, nullptr, &status);
				CHECK_MSTATUS_AND_THROW(status);
			}

			MFnAnimCurve animFn(animCurveObj);

			for (int frame = sF; frame <= eF; ++frame) {
				time.setValue(frame);
				anim.setCurrentTime(time);

				MMatrix mtx = animMatricesMaya[name][frame];
				MTransformationMatrix tm(mtx);
				MVector translate = tm.translation(MSpace::kWorld);
				MEulerRotation euler = tm.rotation().asEulerRotation();

				double value = 0.0;
				if (attrIndex < 3) { // translate
					value = (attrIndex == 0 ? translate.x : (attrIndex == 1 ? translate.y : translate.z));
				}
				else {       // rotate
					double rad = (attrIndex == 3 ? euler.x : (attrIndex == 4 ? euler.y : euler.z));
					value = MAngle(rad).asDegrees();
				}

				animFn.addKey(MTime(frame, currentUnit), value);
			}

		}
	}


	if (bUpdateJointWeight)
	{
		updateResultSkinWeight(skinMeshName);
	}

	MGlobal::displayInfo("Applied animation / updated weights for mesh: " + MString(skinMeshName.c_str()));
}


PYBIND11_MODULE(_core, m) {
	py::class_<DemBonesModel>(m, "DemBones")
		.def(py::init<>())
		.def_readwrite("bind_update", &DemBonesModel::bindUpdate, "Bind transformation update, 0=keep original, 1=set translations to p-norm centroids (using #transAffineNorm) and rotations to identity, 2=do 1 and group joints, default = 0")
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
        .def_readwrite("lock_bone_attr_name", &DemBonesModel::lockBoneAttrName, "Name of the attribute used to lock bone transformations, default = 'demLockBones'")
		.def_readwrite("nBones", &DemBonesModel::nBoneCount, "number of bones")
		.def_readonly("start_frame", &DemBonesModel::sF, "Start frame of solver")
		.def_readonly("end_frame", &DemBonesModel::eF, "End frame of solver")
		.def_readonly("influences", &DemBonesModel::boneName, "List of all influences")
		.def_readonly("weights", &DemBonesModel::weightsMaya, "List of weights for all influences and vertices")
		.def("rmse", &DemBonesModel::rmse, "Root mean squared reconstruction error")
		.def("compute", &DemBonesModel::compute, "Skinning decomposition of alternative updating weights and bone transformations", py::arg("source"), py::arg("target"), py::arg("start_frame"), py::arg("end_frame"))
        .def("apply_animation_and_weights", &DemBonesModel::applyAnimationAndWeights, py::arg("skin_mesh"),py::arg("b_update_joint_weight"), "Apply computed animation and weights to the provided skin mesh")
		.def("bind_matrix", &DemBonesModel::bindMatrix, "Get the bind matrix for the provided influence", py::arg("influence"))
		.def("anim_matrix", &DemBonesModel::animMatrix, "Get the animation matrix for the provided influence at the provided frame", py::arg("influence"), py::arg("frame"))
		.def("update_result_skin_weight", & DemBonesModel::updateResultSkinWeight, py::arg("skin_mesh"))
		;
}
