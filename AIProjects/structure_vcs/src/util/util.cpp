#include <string>
#include "document.h"

namespace util {

std::string urlencode(const std::string &s)
{
    //RFC 3986 section 2.3 Unreserved Characters (January 2005)
    const std::string unreserved = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.~";

    std::string escaped="";
    for(size_t i=0; i<s.length(); i++)
    {
        if (unreserved.find_first_of(s[i]) != std::string::npos)
        {
            escaped.push_back(s[i]);
        }
        else
        {
            escaped.append("%");
            char buf[3];
            sprintf(buf, "%.2X", s[i]);
            escaped.append(buf);
        }
    }
    return escaped;
}

void merge_docs(rapidjson::Value &dstObject, rapidjson::Value &srcObject, rapidjson::Document::AllocatorType &allocator)
{
    for (auto srcIt = srcObject.MemberBegin(); srcIt != srcObject.MemberEnd(); ++srcIt)
    {
        auto dstIt = dstObject.FindMember(srcIt->name);
        if (dstIt != dstObject.MemberEnd())
        {
            assert(srcIt->value.GetType() == dstIt->value.GetType());
            if (srcIt->value.IsArray())
            {
                for (auto arrayIt = srcIt->value.Begin(); arrayIt != srcIt->value.End(); ++arrayIt)
                {
                    dstIt->value.PushBack(*arrayIt, allocator);
                }
            }
            else if (srcIt->value.IsObject())
            {
                merge_docs(dstIt->value, srcIt->value, allocator);
            }
            else
            {
                dstIt->value = srcIt->value;
            }
        }
        else
        {
            dstObject.AddMember(srcIt->name, srcIt->value, allocator);
        }
    }
}

}