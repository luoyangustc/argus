#pragma once

#include <curl/multi.h>
#include <memory>
#include <string>
#include <map>
#include <vector>
#include <list>
using namespace std;
#ifdef _WIN32
#define strcasecmp stricmp
#include <windows.h>
#include <process.h>
#include <winsock2.h>
#include <windows.h>
#include <direct.h>
typedef CRITICAL_SECTION pthread_mutex_t;
#else
#include <pthread.h>
#include <sys/stat.h>
#endif
#include <stdint.h>
#include <string.h>

typedef int (*http_complete_callback)(uint32_t request_id, void* user_p, int err_code, int http_code, const char* http_resp);

enum { RET_OK=0, RET_MALLOC_FAILED, RET_INVALID_PARAMS, RET_FOPEN_FILE_FAILED,
       RET_EMPTY_FILE, RET_CREATE_DL_FILE_FAILED, RET_INIT_SOCKET_FAILED,
       RET_FILE_ALREADY_DOWNLOADING, RET_FILE_ALREADY_UPLOADING, RET_WEB_REQUEST_NOT_INITIALIZED};

struct WebTask;
class LockGuard;
class MemInfo;

class Mutex
{
public:
	Mutex();
	~Mutex();
	void accquire();
	void release();
private:
	pthread_mutex_t m;
};

class WebRequest
{
public:
	WebRequest(void);
	~WebRequest(void);
	static WebRequest& instance(void);
	static void delete_instance(void);
	void start(void);
	void stop(void);
	int HttpDownloadFile(const char* remote_uri, const char* local_save_path, uint32_t* request_id, http_complete_callback func_callback, void* user_p=NULL, uint32_t connect_timeout=10, uint32_t transfer_timeout=0);
	int HttpUploadFile(const char* remote_uri, const char* local_save_path, uint32_t* request_id, http_complete_callback func_callback, void* user_p=NULL, char* post_data=0, uint32_t connect_timeout=10, uint32_t transfer_timeout=0);
	int SubmitHttpRequest(const char* remote_uri, uint32_t* request_id, http_complete_callback func_callback, void* user_p=NULL, unsigned char* post_data=0, uint32_t post_data_len=0, uint32_t connect_timeout=10, uint32_t transfer_timeout=0);
private:
	WebRequest(WebRequest& rhs);
	WebRequest& operator=(const WebRequest& rhs);
private:
	void addWebTask(WebTask* task);
	CURL* createWebRequest(WebTask* task);
	bool getTaskList(std::list<WebTask*>& tasklist);
	int checkRepeatTask(WebTask* task, std::map<long,WebTask*>& taskmap);
#ifdef _WIN32
	static unsigned __stdcall thread_proc(void * args);
#else
	static void* thread_proc(void* args);
#endif
private:
	volatile bool is_stopped_;
#ifdef _WIN32
	HANDLE m_thread;
#else
	pthread_t m_thread;
#endif
	Mutex m_mutex;
	std::list<WebTask*> m_web_task_list;
	unsigned int m_req_id;
	static std::auto_ptr<WebRequest> m_instance;
};

class LockGuard
{
public:
	LockGuard(Mutex& lock);
	~LockGuard();
private:
	Mutex& m_lock;	
};

class MemInfo
{
public:
	MemInfo();
	void push(char*pdata, int data_len);
	void fill(std::string& sOut);
private:
	std::vector<char> buffer;
};

struct WebTask
{
public:
	WebTask(void);
	~WebTask(void);
	uint32_t req_id;
	bool bWriteFile;
	bool bReadFile;
	string uri;
	string savePath;
	unsigned char* postData;
	uint32_t post_data_len;
	FILE* fp;
	http_complete_callback complete_callback;
	MemInfo mem_info;
	curl_slist* headerList;
	curl_httppost* form_post;
	void* user_p;
	uint32_t connect_timeout;
	uint32_t transfer_timeout;
};