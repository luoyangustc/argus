#ifndef _CONFIG_HELPER_
#define _CONFIG_HELPER_

#include <string>
#include <map>
#include <fstream>

using namespace std;

enum EN_CFG_ERR_TYPE
{
	EN_CFG_SUCCESS,
    CFG_ERR_NOSECTION,
    CFG_ERR_NOITEM,
	EN_CFG_ERR_OPENFILE_FAIL
};

#define  EN_CFG_SUCCESS              0
#define  EN_CFG_ERR_NOSECTION       -1
#define  EN_CFG_ERR_NOITEM          -2
#define  EN_CFG_ERR_OPENFILE_FAIL   -3

typedef  map<string, string>     ITEM_MAP;  //key--value
typedef  map<string, ITEM_MAP>   SECTION_MAP;  //section---item

class CConfigHelper
{
public:
	CConfigHelper();
	virtual ~CConfigHelper();
	int read_config_file(const string& filename);
	int save_config_file(const string& filename);
	void set_value(const string& section, const string& key, const string& value);
	void set_value(const string& section, const string& key, unsigned int unvalue);
	int get_section(const string& section, ITEM_MAP &keyValueMap);
	int get_value(string& value, const string& section, const string& key, const string& default_value="");
    int get_value(bool& value, const string& section, const string& key, bool default_value=false);
	int get_value(unsigned int& value, const string& section, const string& key, unsigned int default_value=0);
public:
    static string get_module_short_name();  // abc
    static string get_module_full_name();   // /opt/abc
	static string get_default_config_filename();
    static string get_default_config_dir();
private:
    string m_strFilePath;
	SECTION_MAP m_SectionMap;
};

unsigned int GetPrivateProfileInt(
	const char* szAppName,
	const char* szKeyName,
	int nDefault,
	const char* szFileName
	);

unsigned int GetPrivateProfileString(
	const char* szAppName,
	const char* szKeyName,
	const char* szDefault,
	char* szReturnedString,
	unsigned int nSize,
	const char* szFileName
	);

bool WritePrivateProfileString(
	const char* szAppName,
	const char* szKeyName,
	const char* szString,
	const char* szFileName
	);

#endif