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

namespace MayaDemBoneUtils {

	inline std::string DebugTraceFileLine(const char* file, int line) {
		return std::string(file) + ":" + std::to_string(line);
	}

	#define TRACE_FILE_LINE() DebugTraceFileLine(__FILE__, __LINE__)

	#define LOG(str) {                                              \
		cout << "[" << TRACE_FILE_LINE() << "] ";					\
		cout << str << endl;                                        \
	}

	inline void CheckMStateAndThrow(MStatus& status)
	{
		if (status.statusCode() == MStatus::kSuccess)
			return;

		if (status.error())
		{
			if (status.errorString().length() >= 1) // avoid nothing happen
			{

				cout << "[INFO] " << status.errorString().asUTF8() << endl;
				py::exec("raise RuntimeError ('" + string(status.errorString().asUTF8()) + "')");
			}
			else
			{
				MGlobal::displayInfo(status.errorString());
			}
		}
	}

	#define CHECK_MSTATUS_AND_THROW(aStatus) {															\
		if (aStatus.statusCode() != MStatus::kSuccess) LOG("[DEBUG] Status not success!\n" << endl);	\
		CheckMStateAndThrow(aStatus);																	\
	}

}


namespace Conversion {

	using namespace MayaDemBoneUtils;

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

	template<typename ... Args>
	static std::string FormatString(const std::string& format, Args ... args)
	{
		auto size_buf = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1;
		std::unique_ptr<char[]> buf(new(std::nothrow) char[size_buf]);

		if (!buf)
			return std::string("");

		std::snprintf(buf.get(), size_buf, format.c_str(), args ...);
		return std::string(buf.get(), buf.get() + size_buf - 1);
	}

	inline MString toMString(const string& name) {
		return MString(name.c_str());
    };

	inline string toString(const MString& name) {
		return string(name.asUTF8());
    };

	MDagPath toMDagPath(string& name, bool shape) {
		MStatus status;
		MDagPath dag;
		MSelectionList selection;
		
		status = selection.add(MString(name.c_str()));
		CHECK_MSTATUS_AND_THROW(status);

		status = selection.getDagPath(0, dag, MObject::kNullObj);
		CHECK_MSTATUS_AND_THROW(status);

		if (shape) {
			status = dag.extendToShape();

			if (status.statusCode() != MStatus::kSuccess)
				status = dag.extendToShapeDirectlyBelow(0);
		}
		CHECK_MSTATUS_AND_THROW(status);

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

		matrix = matrix.rotateTo(MEulerRotation(rotate));

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
