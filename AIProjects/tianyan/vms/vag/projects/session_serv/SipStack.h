#ifndef __SIP_STACK_H__
#define __SIP_STACK_H__
#include <vector>
#include <string>
#include <pjsip.h>
#include <pjsip_ua.h>
#include <pjlib-util.h>
#include <pjlib.h>
#include <pjsip_auth.h>
#include "tick.h"

using namespace std;


namespace SipStack
{
    struct sip_hdr
    {
        string head;
        string value;
    };

    struct sip_invite_session
    {
        bool running_;
        pjsip_inv_session* inv;
        tick_t start_tick;
    };

    int sipstk_init(int sip_port, const char* local_addr, const char* log_filename);
    int sipstk_send_manscdp_xml( const char* target, const char* from, const char* to,  const char* contact, const char* xml);
    int sipstk_send_response( int st_code, vector<sip_hdr>* head_list, pjsip_rx_data *rdata);
    int sipstk_send_invite( const char* local_uri, const char* remote_uri, vector<sip_hdr>* head_list, const char* sdp, sip_invite_session* inv_session);
}

#endif //__SIP_STACK_H__
