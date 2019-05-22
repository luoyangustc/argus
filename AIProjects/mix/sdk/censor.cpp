#include "censor.hpp"
#include "http_request.hpp"
#include "http_response.hpp"
#include "json.hpp"
#include <time.h>
#include <iostream>
#include <sstream>
#include <memory>


using namespace qc;
using namespace nlohmann;


static const std::string base64_chars = 
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

std::string base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len) {
  std::string ret;
  int i = 0;
  int j = 0;
  unsigned char char_array_3[3];
  unsigned char char_array_4[4];

  while (in_len--) {
    char_array_3[i++] = *(bytes_to_encode++);
    if (i == 3) {
      char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
      char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
      char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
      char_array_4[3] = char_array_3[2] & 0x3f;

      for(i = 0; (i <4) ; i++)
        ret += base64_chars[char_array_4[i]];
      i = 0;
    }
  }

  if (i)
  {
    for(j = i; j < 3; j++)
      char_array_3[j] = '\0';

    char_array_4[0] = ( char_array_3[0] & 0xfc) >> 2;
    char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
    char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);

    for (j = 0; (j < i + 1); j++)
      ret += base64_chars[char_array_4[j]];

    while((i++ < 3))
      ret += '=';

  }

  return ret;

}

Censor::Censor(){}

Censor::~Censor(){}

void Censor::init(int timeout)
{
    this->timeout = timeout;
    return;
}

int Censor::eval(const void* buf, const int size, const pic_info_t* meta)
{
    if (buf == NULL || size <= 0)
    {
        std::cout << "empty buf, size:" << size <<std::endl;
        return ErrEmptyBuf;
    }
    int ret = 0;
    json jsonReq;
    json jsonMeta;
    std::stringstream out;
    jsonMeta["pid"] = meta->pid;
    jsonMeta["zip_name"] = meta->zip_name;
    jsonMeta["bcp_name"] = meta->bcp_name;
    jsonMeta["data_id"] = meta->data_id;
    jsonMeta["line"] = meta->line;
    jsonMeta["pic_name"] = meta->pic_name;
    jsonReq["uri"] = std::string("data:application/octet-stream;base64,") + base64_encode((unsigned char*)buf, (unsigned int)size);
    jsonReq["meta"] = jsonMeta;
    std::string str = jsonReq.dump();

    UrlRequest req;
    req.timeout={this->timeout,0};
    req.host("127.0.0.1");
    req.port(23400);
    req.uri("/v1/pic");
    req.method("POST");
    req.body(str);
    req.addHeader("Content-Type: application/json;charset=UTF-8");
    auto res = std::move(req.perform());
    if (res.statusCode() == 200)
    {
        json jsonData = json::parse(res.body());
        if (jsonData["skip"] == true) {
            ret = ErrTooManyRequests;
        }
    }
    else
    {
        std::cout << "http error:" << res.statusCode() << std::endl;
        return ErrInternalServerError;
    }
    return ret;
}

metrics_info_t Censor::metrics()
{
    std::stringstream out;
    metrics_info_t info;

    UrlRequest req;
    req.timeout={this->timeout,0};
    req.host("127.0.0.1");
    req.port(23400);
    req.uri("/v1/metrics");
    req.method("GET");
     auto res = std::move(req.perform());

    if (res.statusCode() == 200)
    {
        json jsonData = json::parse(res.body());
        info.total = jsonData["total"];
        info.done = jsonData["done"];
        info.waiting = jsonData["waiting"];
        info.censor = jsonData["censor"];
        info.skip = jsonData["skip"];
        info.error = jsonData["error"];
        info.last_qps = jsonData["last_qps"];
        info.last_filter_rate = jsonData["last_rate"];
        info.qps = jsonData["qps"];
        info.filter_rate = jsonData["rate"];
        info.normal = jsonData["normal"];
        info.pulp = jsonData["pulp"];
        info.terror = jsonData["terror"];
        info.politicion = jsonData["politicion"];
        info.march = jsonData["march"];
        info.text = jsonData["text"];
        info.runtime = jsonData["runtime"];
    }
    return info;
}