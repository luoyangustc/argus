#pragma once

#include <string>
using namespace std;

//对http url某个参数的值进行url编码
string url_encode(const string& str);

//对http url某个参数的值进行url解码
string url_decode(const string& str);
