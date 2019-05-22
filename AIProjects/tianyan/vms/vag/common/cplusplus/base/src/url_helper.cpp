#include <string.h>
#include "url_helper.h"

static string get_http_url_safe_character(unsigned char ch)
{
	//ÌØÊâ×Ö·û  $-_.+!*¡¯(),
	//±£Áô×Ö·û  &/:;=?@

	static const char* ch_list = "$-_.+!*'(),&/:;=?@";

	static char hex[] = "0123456789ABCDEF";

	if (isascii(ch))
	{
		if (strchr(ch_list, ch) != NULL)
		{
			return string(1,ch);
		}
		else if ((ch>='0' && ch<='9') || (ch>='a' && ch<='z') || (ch>='A' && ch<='Z'))
		{
			return string(1,ch);
		}
		else
		{
			string dst;
			unsigned char c = ch;  
			dst += '%';  
			dst += hex[c / 16];  
			dst += hex[c % 16];
			return dst;
		}

	}
	else
	{
		string dst;
		unsigned char c = ch;  
		dst += '%';  
		dst += hex[c / 16];  
		dst += hex[c % 16];
		return dst;
	}
}

std::string url_encode(const string& str)
{
	string dst;
	for (size_t i = 0; i < str.size(); ++i)  
	{  
		unsigned char cc = str[i];  
		dst += get_http_url_safe_character(cc); 
	}  
	return dst;
}

static unsigned char from_hex (unsigned char ch) 
{
	if (ch <= '9' && ch >= '0')
		ch -= '0';
	else if (ch <= 'f' && ch >= 'a')
		ch -= 'a' - 10;
	else if (ch <= 'F' && ch >= 'A')
		ch -= 'A' - 10;
	else 
		ch = 0;
	return ch;
}

std::string url_decode (const std::string& str) 
{
	string result;
	string::size_type i;
	for (i = 0; i < str.size(); ++i)
	{
		if (str[i] == '+')
		{
			result += ' ';
		}
		else if (str[i] == '%' && str.size() > i+2)
		{
			const unsigned char ch1 = from_hex(str[i+1]);
			const unsigned char ch2 = from_hex(str[i+2]);
			const unsigned char ch = (ch1 << 4) | ch2;
			result += ch;
			i += 2;
		}
		else
		{
			result += str[i];
		}
	}
	return result;
}
