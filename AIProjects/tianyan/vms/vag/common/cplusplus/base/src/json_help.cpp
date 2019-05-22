#include "json_help.h"

namespace json_help{
	int getIntValueFromJsonValue(Json::Value& json_object)
	{
		do 
		{
			if (json_object.empty() )
			{
				break;
			}

			if ( json_object.type() == Json::uintValue )
			{
				return json_object.asUInt();
			}

			if ( json_object.type() == Json::intValue )
			{
				return json_object.asInt();
			}			
		} while (false);				
		return 0;
	}

	void getStringValueFromJsonValue(Json::Value& json_object,string& out_str)
	{
		if (!json_object.empty() && json_object.type() == Json::stringValue )
		{
			out_str = json_object.asCString();
		}
	}
}


