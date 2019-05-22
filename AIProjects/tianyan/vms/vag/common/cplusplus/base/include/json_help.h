#ifndef __JSON_HELP_H__
#define __JSON_HELP_H__
#include "json/include/json.h"
#include "json/include/value.h"
#include <string>
using namespace std;
namespace json_help{
	int getIntValueFromJsonValue(Json::Value& json_object);
	void getStringValueFromJsonValue(Json::Value& json_object,string& out_str);
}

#endif //__JSON_HELP_H__

