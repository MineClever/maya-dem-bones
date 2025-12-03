#pragma once

#include <iostream>
#include <maya/MGlobal.h>
#include <maya/MTypes.h>
#include <maya/MObject.h>
#include <maya/MDagPath.h>
#include <maya/MDagPathArray.h>
#include <maya/MAnimControl.h>
#include <maya/MTime.h>
#include <maya/MPlug.h>
#include <maya/MString.h>
#include <maya/MSelectionList.h>
#include <maya/MMatrix.h>
#include <maya/MTransformationMatrix.h>
#include <maya/MColor.h>
#include <maya/MColorArray.h>
#include <maya/MPoint.h>
#include <maya/MPointArray.h>
#include <maya/MQuaternion.h>
#include <maya/MEulerRotation.h>
#include <maya/MFnMesh.h>
#include <maya/MFnDependencyNode.h>
#include <maya/MFnDagNode.h>
#include <maya/MFnSkinCluster.h>
#include <maya/MItDependencyGraph.h>
#include <Eigen/Dense>

#include <Python.h>
#include <pybind11/eval.h>

namespace py = pybind11;
using namespace std;
using namespace Eigen;


#define LOG(str) {                                              \
	cout << str;                                                \
}


static void CheckMStateAndThrow(MStatus& status)
{
	if (status.statusCode() == MStatus::kSuccess)
		return;

	if (status.error())
	{
		if (status.errorString().length() >= 1) // avoid nothing happen
		{
			LOG("[INFO] " << status.errorString().asUTF8() << endl);
			py::exec("raise RuntimeError ('" + string(status.errorString().asUTF8()) + "')");
		}
		else
		{
			MGlobal::displayInfo(status.errorString());
		}
	}
}

#define CHECK_MSTATUS_AND_THROW(aStatus) CheckMStateAndThrow(aStatus)

namespace Conversion {

	inline void RaiseException(const char* aMessage)
	{
		MGlobal::displayError(aMessage);
		py::exec("raise RuntimeError ('" + string(aMessage)+"')");
	}
	

	inline void RaiseException(string& aMessage)
	{
		MGlobal::displayError(aMessage.c_str());
		py::exec("raise RuntimeError ('" + aMessage + "')");
	}

	inline void RaiseException(MString& aMessage)
	{
		MGlobal::displayError(aMessage.asUTF8());
		py::exec("raise RuntimeError ('" + string(aMessage.asUTF8()) + "')");
	}

	MDagPath toMDagPath(string& name, bool shape) {
		MStatus status;
		MDagPath dag;
		MSelectionList selection;
		
		status = selection.add(MString(name.c_str()));
		LOG("[INFO] Get dagpath from string : "<< name.c_str() << endl);

		status = selection.getDagPath(0, dag, MObject::kNullObj);
		CHECK_MSTATUS_AND_THROW(status);

		if (shape) {
			status = dag.extendToShape();

			if (status.statusCode() != MStatus::kSuccess)
				status = dag.extendToShapeDirectlyBelow(0);
		}
		
		CHECK_MSTATUS_AND_THROW(status);
		LOG("[INFO] Get dagpath -> " << dag.fullPathName().asUTF8() << endl);
		return dag;
	};

	Matrix4d toMatrix4D(MMatrix& source) {
		Matrix4d target;

		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				target(j, i) = source(i, j);

		return target;
	};

	MMatrix toMMatrix(MVector& translate, MVector& rotate, MTransformationMatrix::RotationOrder rotateOrder) {
		MStatus status;
		MTransformationMatrix matrix;
		
		status = matrix.setTranslation(translate, MSpace::kObject);
		CHECK_MSTATUS_AND_THROW(status);

		const double rotation[3] = { rotate.x, rotate.y, rotate.z };
		status = matrix.setRotation(rotation, rotateOrder, MSpace::kObject);
		CHECK_MSTATUS_AND_THROW(status);
		return matrix.asMatrix();
	};

	array<double, 16> toMatrixArray(MMatrix matrix) {
		array<double, 16> matrixArray;

		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				matrixArray[i * 4 + j] = matrix(i, j);

		return matrixArray;
	};

	double toGrayScale(const MColor& c) {
		return 0.2989*c.r + 0.5870*c.g + 0.1140*c.b;
	}

	MDoubleArray toMDoubleArray(const std::vector<double>& vec) {
		return MDoubleArray(vec.data(), static_cast<unsigned int>(vec.size()));
	}
}
