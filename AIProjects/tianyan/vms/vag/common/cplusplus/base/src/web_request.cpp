#include <unistd.h>
#include "web_request.h"
//#include "logging_posix.h"

std::auto_ptr<WebRequest> WebRequest::m_instance;

static size_t readData(char *buffer, size_t size, size_t nmemb, void *user_p);
static size_t writeData(char *buffer, size_t size, size_t nmemb, void *user_p);
static size_t writeBuffer(char *buffer, size_t size, size_t nmemb, void *user_p);
static void addDefaultHttpHeader(CURL *eh, WebTask* pTask);

static void create_prefix_dir(string filepath);
static string urlencode(const char* url);
static void split(const std::string& s, const std::string& delim, std::vector<std::string>& elems);
static void wait_ms(int ms);

WebRequest::WebRequest():is_stopped_(true),m_req_id(0)
{
}

WebRequest::~WebRequest()
{
}

WebRequest& WebRequest::instance()
{
	if (m_instance.get() == NULL)
	{
		m_instance = auto_ptr<WebRequest> (new WebRequest);
	}
	return *m_instance;
}

void WebRequest::delete_instance(void)
{
	if ( m_instance.get() != NULL )
	{
		delete m_instance.release();
	}
}

void WebRequest::start()
{
	if (is_stopped_)
	{
#ifdef _WIN32
		m_thread = (HANDLE)_beginthreadex( NULL, 0, thread_proc, this, 0, NULL);
#else
		pthread_create(&m_thread, NULL, thread_proc, this); 
#endif
		is_stopped_ = false;
	}
}

void WebRequest::stop()
{
	if ( ! is_stopped_)
	{
		is_stopped_ = true;
#ifdef _WIN32
		WaitForSingleObject(m_thread, INFINITE);
		CloseHandle(m_thread);
#else
		pthread_join(m_thread, NULL);
#endif
	}	
}

int WebRequest::HttpDownloadFile(const char* remote_uri, const char* local_save_path, uint32_t* request_id, http_complete_callback func_callback, void* user_p, uint32_t connect_timeout, uint32_t transfer_timeout)
{
	WebTask* web_task = NULL;

	do
	{
		LockGuard guard(m_mutex);
		if (is_stopped_)
			return RET_WEB_REQUEST_NOT_INITIALIZED;

		if (remote_uri==NULL || local_save_path==NULL || strlen(remote_uri)==0 || strlen(local_save_path)==0)
			return RET_INVALID_PARAMS;

		web_task = new WebTask;
		if (web_task == NULL)
			return RET_MALLOC_FAILED;

		m_req_id++;
		*request_id = m_req_id;

		web_task->req_id = m_req_id;
		web_task->bWriteFile = true;
		web_task->uri = remote_uri;
		web_task->savePath = local_save_path;
		web_task->postData = NULL;
		web_task->complete_callback = func_callback;
		web_task->user_p = user_p;
		web_task->connect_timeout = connect_timeout;
		web_task->transfer_timeout = transfer_timeout;

		WebRequest::instance().addWebTask(web_task);
	}
	while(false);

	return 0;
}

int WebRequest::HttpUploadFile(const char* remote_uri, const char* local_save_path, uint32_t* request_id, http_complete_callback func_callback, void* user_p, char* post_data, uint32_t connect_timeout, uint32_t transfer_timeout)
{
	WebTask* web_task = NULL;

	do
	{
		LockGuard guard(m_mutex);
		if (is_stopped_)
			return RET_WEB_REQUEST_NOT_INITIALIZED;

		if (remote_uri==NULL || local_save_path==NULL || strlen(remote_uri)==0 || strlen(local_save_path)==0)
			return RET_INVALID_PARAMS;

		web_task = new WebTask;
		if (web_task == NULL)
			return RET_MALLOC_FAILED;

		m_req_id++;
		*request_id = m_req_id;

		web_task->req_id = m_req_id;
		web_task->bReadFile = true;
		web_task->uri = remote_uri;
		web_task->savePath = local_save_path;
		web_task->postData = (unsigned char*)post_data;
		web_task->complete_callback = func_callback;
		web_task->user_p = user_p;
		web_task->connect_timeout = connect_timeout;
		web_task->transfer_timeout = transfer_timeout;

		WebRequest::instance().addWebTask(web_task);
	}
	while(false);

	return 0;
}

int WebRequest::SubmitHttpRequest(const char* remote_uri, uint32_t* request_id, http_complete_callback func_callback, void* user_p, unsigned char* post_data, uint32_t post_data_len, uint32_t connect_timeout, uint32_t transfer_timeout)
{
	WebTask* web_task = NULL;

	do
	{
		LockGuard guard(m_mutex);
		if (is_stopped_)
			return RET_WEB_REQUEST_NOT_INITIALIZED;

		if (remote_uri==NULL || strlen(remote_uri)==0 || (post_data!=NULL && post_data_len==0))
			return RET_INVALID_PARAMS;

		web_task = new WebTask;
		if (web_task == NULL)
			return RET_MALLOC_FAILED;

		m_req_id++;
		*request_id = m_req_id;

		web_task->req_id = m_req_id;
		web_task->bWriteFile = false;
		web_task->bReadFile = false;
		web_task->uri = remote_uri;
		web_task->postData = post_data;
		web_task->post_data_len = post_data_len;
		web_task->complete_callback = func_callback;
		web_task->user_p = user_p;
		web_task->connect_timeout = connect_timeout;
		web_task->transfer_timeout = transfer_timeout;

		WebRequest::instance().addWebTask(web_task);
	}
	while(false);

	return 0;
}

void WebRequest::addWebTask(WebTask* task)
{
	this->m_web_task_list.push_back(task);
}

bool WebRequest::getTaskList(std::list<WebTask*>& tasklist)
{
	LockGuard guard(m_mutex);
	bool bRet = m_web_task_list.empty() ? false : true;
	if (bRet)
	{
		tasklist.clear();
		tasklist.assign(m_web_task_list.begin(),m_web_task_list.end());
		m_web_task_list.clear();
	}

	return bRet;
}

CURL* WebRequest::createWebRequest(WebTask* task)
{
	CURL *eh = curl_easy_init();
	if (eh == NULL)
	{
		task->complete_callback(task->req_id, task->user_p, RET_INIT_SOCKET_FAILED, -1, NULL);
		delete task;
		return NULL;
	}
	
	task->uri = urlencode(task->uri.c_str());
	curl_easy_setopt(eh, CURLOPT_URL, task->uri.c_str());
	curl_easy_setopt(eh, CURLOPT_FOLLOWLOCATION, 1); //support http 301,302
	curl_easy_setopt(eh, CURLOPT_FORBID_REUSE, 1);//禁止复用TCP连接
	if (task->connect_timeout > 0)
		curl_easy_setopt(eh, CURLOPT_CONNECTTIMEOUT, task->connect_timeout);

	if (task->transfer_timeout > 0)
		curl_easy_setopt(eh, CURLOPT_TIMEOUT, task->transfer_timeout);

	if (task->bWriteFile)
	{
		curl_easy_setopt(eh, CURLOPT_WRITEFUNCTION, &writeData);	
		curl_easy_setopt(eh, CURLOPT_WRITEDATA, task->fp);
	}
	else if (task->bReadFile)
	{
		struct curl_httppost *formpost = 0;
		struct curl_httppost *lastptr  = 0;
		curl_formadd(&formpost, &lastptr, CURLFORM_PTRNAME, "file", CURLFORM_FILE, task->savePath.c_str(), CURLFORM_END);
		
		if (task->postData != NULL) //form params list
		{
			vector<string> elems;
			split((char*)task->postData, "&", elems);
			for(vector<string>::iterator it=elems.begin(); it!=elems.end(); it++)
			{
				vector<string> tmp;
				split(*it, "=", tmp);
				if (tmp.size() == 2)
				{
					curl_formadd(&formpost, &lastptr, CURLFORM_COPYNAME, tmp[0].c_str(), CURLFORM_COPYCONTENTS, tmp[1].c_str(), CURLFORM_END);
				}
			}
		}
		
		curl_easy_setopt(eh, CURLOPT_HTTPPOST, formpost);
		curl_easy_setopt(eh, CURLOPT_WRITEFUNCTION, &writeBuffer);
		curl_easy_setopt(eh, CURLOPT_WRITEDATA, &task->mem_info);
		
		task->form_post = formpost;
	}
	else
	{
		if (task->postData != NULL) //contain POST form
		{
			curl_easy_setopt(eh, CURLOPT_POST, 1);
			curl_easy_setopt(eh, CURLOPT_POSTFIELDS, task->postData);
			curl_easy_setopt(eh, CURLOPT_POSTFIELDSIZE, task->post_data_len);
		}
		
		curl_easy_setopt(eh, CURLOPT_WRITEFUNCTION, &writeBuffer);
		curl_easy_setopt(eh, CURLOPT_WRITEDATA, &task->mem_info);
	}
	
	addDefaultHttpHeader(eh, task);

	return eh;
}

int WebRequest::checkRepeatTask(WebTask* task, std::map<long,WebTask*>& taskmap)
{
	bool bRepeat = false;
	if (task->bWriteFile || task->bReadFile)
	{
		for(std::map<long,WebTask*>::const_iterator iter=taskmap.begin(); iter!=taskmap.end(); iter++)
		{
			if (strcasecmp(task->savePath.c_str(), iter->second->savePath.c_str()) == 0)
			{
				bRepeat = true;
				break;
			}
		}

		if (bRepeat)
		{
			int code = task->bWriteFile ? RET_FILE_ALREADY_DOWNLOADING : RET_FILE_ALREADY_UPLOADING;
			task->complete_callback(task->req_id, task->user_p, code, -1, NULL);
			delete task;
			return 1;
		}

		if (task->bWriteFile)
		{
			create_prefix_dir(task->savePath);
		}

		if (task->bWriteFile)
			task->fp = fopen(task->savePath.c_str(), "wb");
		/*else
			task->fp = fopen(task->savePath.c_str(), "rb");*/

		if (task->bWriteFile && task->fp == NULL)
		{
			int code = task->bWriteFile ? RET_CREATE_DL_FILE_FAILED : RET_FOPEN_FILE_FAILED;
			task->complete_callback(task->req_id, task->user_p, code, -1, NULL);
			delete task;
			return 2;
		}
	}

	return 0;
}

#ifdef _WIN32
unsigned __stdcall WebRequest::thread_proc(void * args)
#else
void* WebRequest::thread_proc(void* args)
#endif
{
	curl_global_init(CURL_GLOBAL_ALL);
	CURLM* multi_easy_list = curl_multi_init();

	WebRequest* pArgs = (WebRequest*)args;
	std::map<long,WebTask*> run_tasks;

	while( ! pArgs->is_stopped_)
	{
        std::list<WebTask*> new_tasks;
		bool has_newtask = pArgs->getTaskList(new_tasks);
        if(has_newtask)
        {
            std::list<WebTask*>::iterator it=new_tasks.begin();
            for( ; it!=new_tasks.end(); it++)
            {
                if (pArgs->checkRepeatTask((*it), run_tasks) != 0)
                {
                    WebTask* pTask = *it;
                    printf("repeat tast, req_id(%u), url(%s)\n",pTask->req_id, pTask->uri.c_str());
                    continue;
                }

                CURL* eh = pArgs->createWebRequest(*it);
                if (eh == NULL)
                {
                    WebTask* pTask = *it;
                    printf("create WebRequest objs failed, req_id(%u), url(%s)\n",pTask->req_id, pTask->uri.c_str());
                    continue;
                }

                curl_multi_add_handle(multi_easy_list, eh);
                run_tasks.insert(make_pair((long)eh, *it));
                {
                    WebTask* pTask = *it;
                    printf("add task map, req_id(%u), url(%s)\n",pTask->req_id, pTask->uri.c_str());
                }
            }
        }

        if( run_tasks.empty() )
        {
            wait_ms(20);
            continue;
        }

        printf( "start-->run_tasks size(%u)\n", (unsigned int)run_tasks.size() );

        bool is_err = false;
        int repeats = 0;
        int still_running = 0;
        do {
            CURLMcode mc;
            int numfds;

            mc = curl_multi_perform(multi_easy_list, &still_running);
            if ( mc != CURLM_OK )
            {
                printf("curl_multi_perform fail, code(%d)\n",(int)mc);
                is_err = true;
                break;
            }

            // wait for activity, timeout or "nothing"
            mc = curl_multi_wait(multi_easy_list, NULL, 0, 1000, &numfds);
            if ( mc != CURLM_OK )
            {
                printf("curl_multi_wait fail, code(%d)\n", (int)mc);
                is_err = true;
                break;
            }

            /* 'numfds' being zero means either a timeout or no file wait for.
               Try timeout on first occurrence, then assume descriptors and no file
               descriptors to wait for means milliseconds. */
            if ( !numfds )
            {
                repeats++; // count number of repeated zero numfds.
                if(repeats > 1)
                {
                    wait_ms(100);
                }
            }
            else
            {
                repeats = 0;
            }
        } while (still_running);

        if(is_err)
        {
            wait_ms(20);
            continue;
        }

        int msgs_left = 0;
        CURLMsg* pMsg = NULL;
        while((pMsg = curl_multi_info_read(multi_easy_list, &msgs_left)))
        {
            if (pMsg->msg != CURLMSG_DONE)
                continue;

            CURL* eh = pMsg->easy_handle;
            CURLcode return_code = pMsg->data.result;
            int http_status_code = 0;
            if(return_code == CURLE_OK)
                curl_easy_getinfo(eh, CURLINFO_RESPONSE_CODE, &http_status_code);

            std::map<long,WebTask*>::iterator iter = run_tasks.find((long)eh);
            WebTask* pTask = iter->second;
            if (pTask->bWriteFile || pTask->bReadFile)
            {
                FILE* fp = pTask->fp;
                if (fp != NULL)
                    fclose(fp);
            }

            uint32_t req_id = pTask->req_id;
            curl_multi_remove_handle(multi_easy_list, eh);
            curl_easy_cleanup(eh);
            run_tasks.erase(iter);
            printf("delete task map, req_id(%u), url(%s)\n",pTask->req_id, pTask->uri.c_str());

            const char* http_resp = "";
            std::string http_resp_str;
            if ( ! pTask->bWriteFile)
            {
                pTask->mem_info.fill(http_resp_str);
                http_resp = http_resp_str.c_str();
            }
            pTask->complete_callback(
                        req_id, 
                        pTask->user_p, 
                        (int)((return_code==CURLE_OK)?RET_OK:return_code), 
                        http_status_code, 
                        http_resp );
            printf("req_id=%u,http_code=%d,uri=%s,resp=%s\n",
                pTask->req_id,http_status_code,pTask->uri.c_str(),http_resp);
            delete pTask;
        }
		
        printf( "end-->run_tasks size(%u)\n", (unsigned int)run_tasks.size() );
	}

	if (multi_easy_list != NULL)
	{
		curl_multi_cleanup(multi_easy_list);
		multi_easy_list = NULL;
	}

	curl_global_cleanup();

	return 0;
}

static size_t readData(char *buffer, size_t size, size_t nmemb, void *user_p)
{
	FILE *fp = (FILE *)user_p;
	return fread(buffer, size, nmemb, fp);
}

static size_t writeData(char *buffer, size_t size, size_t nmemb, void *user_p)
{
	FILE *fp = (FILE *)user_p;
	return fwrite(buffer, size, nmemb, fp);
}

static size_t writeBuffer(char *buffer, size_t size, size_t nmemb, void *user_p)
{
	MemInfo* pmem = (MemInfo*)user_p;
	int recv_size = size * nmemb;
	pmem->push(buffer, recv_size);
	return recv_size;
}

static void addDefaultHttpHeader(CURL *eh, WebTask* pTask)
{
	map<string,string> header;
	//header.insert(make_pair("Content-Type","application/x-www-form-urlencoded"));
	//header.insert(make_pair("User-Agent","Mozilla/4.0"));
	
	int i = 0;
	curl_slist *plist;
	for(map<string,string>::iterator it=header.begin(); it!=header.end(); it++)
	{
		string pair;
		pair += it->first;
		pair += ":";
		pair += it->second;
		if (i == 0)
			plist = curl_slist_append(NULL,pair.c_str());
		else
			curl_slist_append(plist,pair.c_str());
		i++;
	}
	
	if (header.size() > 0)
	{
		curl_easy_setopt(eh, CURLOPT_HTTPHEADER, plist);
		pTask->headerList = plist;
	}
}

/****************************************************************************************************************************************************/

Mutex::Mutex()
{
#ifdef _WIN32
		InitializeCriticalSection(&m);
#else
		pthread_mutex_init(&m, NULL);
#endif
}

Mutex::~Mutex()
{
#ifdef _WIN32
		DeleteCriticalSection(&m);
#else
		pthread_mutex_destroy(&m);
#endif
}

void Mutex::accquire()
{
#ifdef _WIN32
		EnterCriticalSection(&m);
#else
		pthread_mutex_lock(&m);
#endif
}

void Mutex::release()
{
#ifdef _WIN32
		LeaveCriticalSection(&m);
#else
		pthread_mutex_unlock(&m);
#endif
}


LockGuard::LockGuard(Mutex& lock):m_lock(lock)
{
		m_lock.accquire();
}

LockGuard::~LockGuard()
{
		m_lock.release();
}

MemInfo::MemInfo()
{	
	buffer.reserve(8192);
}
	
void MemInfo::push(char*pdata, int data_len)
{
		char* p = pdata;
		for(int i=0; i<data_len; i++)
			buffer.push_back(*p++);
}

void MemInfo::fill(std::string& sOut)
{
		sOut.clear();
		sOut.assign(buffer.begin(),buffer.end());
}

WebTask::WebTask(void) 
{ 
		bWriteFile = false;
		bReadFile = false;
		postData=NULL; 
		fp=NULL; 
		post_data_len=0; 
		headerList=NULL;
		form_post=NULL; 
		user_p = NULL;
}

WebTask::~WebTask(void) 
{ 
		if (postData != NULL)
			delete[] postData;
		if (headerList != NULL)
			curl_slist_free_all(headerList);
		if (form_post != NULL)
			curl_formfree(form_post);	
}

static void create_prefix_dir(string filepath)
{
#ifdef _WIN32
	char c = '\\';
#else
	char c = '/';
#endif

	string sPath = filepath;
	if (sPath.empty())
		return;

	if (sPath[sPath.size()-1] != c)
		sPath += c;

	for(std::string::iterator it=sPath.begin(); it!=sPath.end(); it++)
	{
		if ((*it) == c)
		{
			string tmp;
			tmp.assign(sPath.begin(), it);
			if (access(tmp.c_str(), 0) != 0)
			{
#ifdef _WIN32
				_mkdir(tmp.c_str());
#else
				mkdir(tmp.c_str(), 0755);
#endif
			}
		}
	}
}

static string get_http_url_safe_character(unsigned char ch)
{
	//特殊字符  $-_.+!*’(),
	//保留字符  &/:;=?@
	
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

static string form_value_encode(const string& src)
{
	string dst;
	for (size_t i = 0; i < src.size(); ++i)  
	{  
		unsigned char cc = src[i];  
		dst += get_http_url_safe_character(cc); 
	}  
	return dst;
}

static string urlencode(const char* url)
{
	const char* pos = strstr(url,"?");
	if (pos == NULL)
		return url;

	std::string prefix = std::string(url,pos);
	if (strlen(pos) == 1)
		return url;

	string src = pos + 1;
	string dst = form_value_encode(src);  

	std::string final_str = prefix + "?";
	final_str += dst;
	return final_str;
}

static void split(const std::string& s, const std::string& delim, std::vector<std::string>& elems)
{
	elems.clear();
    size_t pos = 0;
    size_t len = s.length();
    size_t delim_len = delim.length();
    if (delim_len == 0) return;
    while (pos < len)
    {
        int find_pos = s.find(delim, pos);
        if (find_pos < 0)
        {
			string item = s.substr(pos, len - pos);
			if (item.size() > 0)
				elems.push_back(item);
            break;
        }

		string item = s.substr(pos, find_pos - pos);
		if (item.size() > 0)
			elems.push_back(item);
        pos = find_pos + delim_len;
    }
}

static void wait_ms(int ms)
{
#ifdef _WIN32
    Sleep(ms);
#else
    usleep(ms * 1000);
#endif
}