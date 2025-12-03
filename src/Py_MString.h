
#pragma once

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <maya/MString.h>
#include <maya/MUniqueString.h>

namespace Py_MString_private
{
	// Adapts MString* to the required API to use string_caster
	template<typename StringType>
	class Py_StringCastAdapter
	{
	public:
		using value_type = char;
		Py_StringCastAdapter() {}
		Py_StringCastAdapter(const char* aBuffer, size_t)
			: myData(aBuffer)
		{
		}

		Py_StringCastAdapter(const Py_StringCastAdapter& anOther)
			: myData(anOther.myData)
		{
		}

		Py_StringCastAdapter(Py_StringCastAdapter&& anOther)
			: myData(std::move(anOther.myData))
		{
		}

		Py_StringCastAdapter& operator=(const Py_StringCastAdapter& anOther)
		{
			myData = anOther.myData;
			return *this;
		}

		Py_StringCastAdapter& operator=(Py_StringCastAdapter&& anOther)
		{
			myData = std::move(anOther.myData);
			return *this;
		}
		Py_StringCastAdapter(const StringType& anOther)
			: myData(anOther)
		{
		}
		Py_StringCastAdapter(StringType&& anOther)
			: myData(anOther)
		{
		}

		size_t size() const { return myData.numChars(); }
		const char* data() const { return myData.asChar(); }
		StringType myData;
	};
} // namespace Py_MString_private

namespace pybind11
{
	namespace detail
	{
		template<>
		struct type_caster<MString> : string_caster<Py_MString_private::Py_StringCastAdapter<MString>>
		{
			operator MString* () { return &this->value.myData; }
			operator MString& () { return this->value.myData; }
			operator MString && () { return std::move(this->value.myData); }
		};

		template<>
		struct type_caster<MUniqueString> : string_caster<Py_MString_private::Py_StringCastAdapter<MUniqueString>>
		{
			operator MUniqueString* () { return &this->value.myData; }
			operator MUniqueString& () { return this->value.myData; }
			operator MUniqueString && () { return std::move(this->value.myData); }
		};
	} // namespace detail
} // namespace pybind11
