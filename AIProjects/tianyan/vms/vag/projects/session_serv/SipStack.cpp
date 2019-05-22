#include <stdio.h>
#include "SipStack.h"
#include "ServerLogical.h"

using namespace SipStack;

static struct sipstk_obj
{
    pj_bool_t       running;
    int             sip_port;
    pj_str_t        local_addr;
    pj_str_t        local_uri;
    pj_str_t        local_contact;
    pj_str_t        realm;
    pjsip_auth_srv  auth_srv;

    int             log_level;
    pj_str_t        log_filename;
    pj_caching_pool cp;
    pj_pool_t       *pool;

    pjsip_endpoint  *sip_endpt;
    unsigned		thread_count;
    pj_thread_t		*sip_thread[32];
} sipstk_obj_;

static int          sip_worker_thread(void *arg);    /* Worker thread for SIP */
static pj_status_t  auth_lookup_cred( pj_pool_t *pool, const pj_str_t *realm, const pj_str_t *acc_name, pjsip_cred_info *cred_info );

static pj_bool_t    on_rx_request( pjsip_rx_data *rdata );
static void         call_on_media_update( pjsip_inv_session *inv, pj_status_t status);
static void         call_on_state_changed( pjsip_inv_session *inv, pjsip_event *e);
static void         call_on_forked(pjsip_inv_session *inv, pjsip_event *e);

static pjsip_module sipstk_mod_ = {
    NULL, NULL,                     /* prev, next. */
    { "mod-session", 13 },          /* Name. */
    -1,                             /* Id */
    PJSIP_MOD_PRIORITY_APPLICATION, /* Priority */
    NULL,                           /* load() */
    NULL,                           /* start() */
    NULL,                           /* stop() */
    NULL,                           /* unload()			*/
    &on_rx_request,                 /* on_rx_request()		*/
    NULL,                           /* on_rx_response()		*/
    NULL,                           /* on_tx_request.		*/
    NULL,                           /* on_tx_response()		*/
    NULL,                           /* on_tsx_state()		*/
};

int sipstk_init(int sip_port, const char* local_addr, const char* log_filename)
{
    pj_status_t status;

    /* Must init PJLIB first */
    status = pj_init();
    if (status != PJ_SUCCESS)
    {
        return -1;
    }

    /* init PJLIB-UTIL: */
    status = pjlib_util_init();
    if(status != PJ_SUCCESS)
    {
        return -1;
    }

    /* Must create a pool factory before we can allocate any memory. */
    pj_caching_pool_init(&sipstk_obj_.cp, &pj_pool_factory_default_policy, 0);

    /* Create application pool for misc. */
    sipstk_obj_.pool = pj_pool_create(&sipstk_obj_.cp.factory, "sip_stk", 4000, 4000, NULL);


    sipstk_obj_.sip_port = sip_port;

    if(!local_addr)
    {
        char ip_addr[PJ_INET_ADDRSTRLEN]={0};
        const pj_str_t *hostname;
        pj_sockaddr_in tmp_addr;
        hostname = pj_gethostname();
        pj_inet_ntop(pj_AF_INET(), &tmp_addr.sin_addr, ip_addr, sizeof(ip_addr));
        sipstk_obj_.local_addr = pj_strdup3(sipstk_obj_.pool, ip_addr);
    }
    else
    {
        sipstk_obj_.local_addr = pj_strdup3(sipstk_obj_.pool, local_addr);
    }

    /* Build local URI and contact */
    char local_uri[64]={0};
    pj_ansi_sprintf( local_uri, "sip:%s:%d", sipstk_obj_.local_addr.ptr, sipstk_obj_.sip_port);
    sipstk_obj_.local_uri = pj_strdup3(sipstk_obj_.pool, local_uri);
    sipstk_obj_.local_contact = pj_strdup3(sipstk_obj_.pool, local_uri);

    /* Create the endpoint: */
    status = pjsip_endpt_create(&sipstk_obj_.cp.factory, sipstk_obj_.local_addr.ptr, &sipstk_obj_.sip_endpt);
    if(status != PJ_SUCCESS)
    {
        return -2;
    }

    sipstk_obj_.log_level = 5;
    sipstk_obj_.log_filename = pj_strdup3(sipstk_obj_.pool, log_filename);

    /* Add UDP transport. */
    {
	    pj_sockaddr_in addr;
	    pjsip_host_port addrname;
	    pjsip_transport *tp;

	    pj_bzero(&addr, sizeof(addr));
	    addr.sin_family = pj_AF_INET();
	    addr.sin_addr.s_addr = 0;
	    addr.sin_port = pj_htons((pj_uint16_t)sipstk_obj_.sip_port);

	    if (sipstk_obj_.local_addr.slen) 
        {

	        addrname.host = sipstk_obj_.local_addr;
	        addrname.port = sipstk_obj_.sip_port;

	        status = pj_sockaddr_in_init(&addr, &sipstk_obj_.local_addr, (pj_uint16_t)sipstk_obj_.sip_port);
	        if (status != PJ_SUCCESS)
            {
		        return -3;
	        }
	    }

	    status = pjsip_udp_transport_start( sipstk_obj_.sip_endpt, &addr, (sipstk_obj_.local_addr.slen ? &addrname:NULL), 1, &tp);
	    if (status != PJ_SUCCESS)
        {
	        return -4;
	    }
    }

    /* 
     * Init transaction layer.
     * This will create/initialize transaction hash tables etc.
     */
    status = pjsip_tsx_layer_init_module(sipstk_obj_.sip_endpt);
    if(status != PJ_SUCCESS)
    {
        return -5;
    }

    /*  Initialize UA layer. */
    status = pjsip_ua_init_module( sipstk_obj_.sip_endpt, NULL );
    if(status != PJ_SUCCESS)
    {
        return -6;
    }

    /* Initialize 100rel support */
    status = pjsip_100rel_init_module(sipstk_obj_.sip_endpt);
    if(status != PJ_SUCCESS)
    {
        return -7;
    }

    /*  Init invite session module. */
    {
	    pjsip_inv_callback inv_cb;

	    /* Init the callback for INVITE session: */
	    pj_bzero(&inv_cb, sizeof(inv_cb));
	    inv_cb.on_state_changed = &call_on_state_changed;
	    inv_cb.on_new_session = &call_on_forked;
	    inv_cb.on_media_update = &call_on_media_update;

	    /* Initialize invite session module:  */
	    status = pjsip_inv_usage_init(sipstk_obj_.sip_endpt, &inv_cb);
        if(status != PJ_SUCCESS)
        {
            return -8;
        }
    }

    /* Register our module to receive incoming requests. */
    status = pjsip_endpt_register_module( sipstk_obj_.sip_endpt, &sipstk_mod_ );
    if(status != PJ_SUCCESS)
    {
        return -9;
    }
    
    status = pjsip_auth_srv_init(sipstk_obj_.pool, &sipstk_obj_.auth_srv, &sipstk_obj_.realm, auth_lookup_cred, 0);
    if(status != PJ_SUCCESS)
    {
        return -10;
    }

    sipstk_obj_.running = PJ_TRUE;
    sipstk_obj_.thread_count = 8;

    for (int i=0; i<sipstk_obj_.thread_count; ++i)
    {
        pj_thread_create( sipstk_obj_.pool, "sip_stack", &sip_worker_thread, NULL, 0, 0, &sipstk_obj_.sip_thread[i]);
    }

    return 0;
}

int sipstk_send_msg_manscdp_xml( const char* target, const char* from, const char* to,  const char* contact, const char* xml)
{
    pj_status_t status = 0;
    const pjsip_method method = { PJSIP_OTHER_METHOD,{ "MESSAGE", 7 } };
    pjsip_tx_data *tdata;
    
    status = pjsip_endpt_create_request(sipstk_obj_.sip_endpt, &method, 
                                &pj_str((char*)target), 
                                &pj_str((char*)from), 
                                &pj_str((char*)to), 
                                &pj_str((char*)contact),
                                NULL, -1,
                                &pj_str((char*)xml), &tdata );
    if(status != PJ_SUCCESS)
    {
        return -1;
    }
    tdata->msg->body->content_type.type = pj_str("Application");
    tdata->msg->body->content_type.subtype = pj_str("MANSCDP+xml");

    status = pjsip_endpt_send_request(sipstk_obj_.sip_endpt, tdata, -1, NULL, NULL);
    if(status != PJ_SUCCESS)
    {
        return -2;
    }

    return 0;
}

int sipstk_send_response( int st_code, vector<sip_hdr>* head_list, pjsip_rx_data *rdata)
{
    pj_status_t status = 0;
    pjsip_tx_data* tdata;
    status = pjsip_endpt_create_response(sipstk_obj_.sip_endpt, rdata, st_code, NULL, &tdata);
    if(status != PJ_SUCCESS)
    {
        return -1;
    }

    if(head_list)
    {
        vector<sip_hdr>::iterator it = head_list->begin();
        for( ; it!=head_list->end(); ++it )
        {
            pj_str_t name = pj_str((char*)it->head.c_str());
            pj_str_t value = pj_str((char*)it->value.c_str());
            pjsip_hdr* hdr = (pjsip_hdr*)pjsip_generic_string_hdr_create(sipstk_obj_.pool, &name, &value);
            pjsip_msg_add_hdr(tdata->msg, hdr);
        }
    }

    if(st_code == 401)
    {
        status = pjsip_auth_srv_challenge(&sipstk_obj_.auth_srv, NULL, NULL, NULL, PJ_FALSE, tdata);
        if(status != PJ_SUCCESS)
        {
            return -2;
        }
    }
    
    pjsip_response_addr addr;
    pjsip_get_response_addr(sipstk_obj_.pool, rdata, &addr);
    status = pjsip_endpt_send_response(sipstk_obj_.sip_endpt, &addr, tdata, NULL, NULL);
    if(status != PJ_SUCCESS)
    {
        return -3;
    }

    return 0;
}

int sipstk_send_invite( const char* local_uri, const char* remote_uri, vector<sip_hdr>* head_list, const char* sdp, sip_invite_session* inv_session)
{
    pj_status_t status = 0;
    pjsip_dialog *dlg;

    status = pjsip_dlg_create_uac( pjsip_ua_instance(), 
                        &sipstk_obj_.local_uri,     /* local URI	    */
                        &sipstk_obj_.local_contact,	/* local Contact    */
                        &pj_str((char*)remote_uri), /* remote URI	    */
                        &pj_str((char*)remote_uri),  /* remote target    */
                        &dlg);		                /* dialog	    */
    if (status != PJ_SUCCESS)
    {
        return -1;
    }

    status = pjsip_inv_create_uac( dlg, NULL, 0, &inv_session->inv);
    if (status != PJ_SUCCESS)
    {
        return -2;
    }

    inv_session->inv->mod_data[sipstk_mod_.id] = inv_session;
    inv_session->start_tick = get_current_tick();

    pjsip_tx_data *tdata;
    status = pjsip_inv_invite( inv_session->inv, &tdata);
    if (status != PJ_SUCCESS)
    {
        return -3;
    }

    if(head_list)
    {
        vector<sip_hdr>::iterator it = head_list->begin();
        for( ; it!=head_list->end(); ++it )
        {
            pj_str_t name = pj_str((char*)it->head.c_str());
            pj_str_t value = pj_str((char*)it->value.c_str());
            pjsip_hdr* hdr = (pjsip_hdr*)pjsip_generic_string_hdr_create(sipstk_obj_.pool, &name, &value);
            pjsip_msg_add_hdr(tdata->msg, hdr);
        }
    }

    if( sdp )
    {
        pjsip_media_type type;
        type.type = pj_str("application");
        type.subtype = pj_str("sdp");

        pj_str_t text = pj_str((char *)sdp);

        tdata->msg->body = pjsip_msg_body_create(sipstk_obj_.pool, &type.type, &type.subtype, &text);
    }
    
    status = pjsip_inv_send_msg(inv_session->inv, tdata);
    if (status != PJ_SUCCESS)
    {
        return -4;
    }

    return 0;
}

static int sip_worker_thread(void *arg)
{
    PJ_UNUSED_ARG(arg);

    while (!sipstk_obj_.running)
    {
        pj_time_val timeout = {0, 10};
        pjsip_endpt_handle_events(sipstk_obj_.sip_endpt, &timeout);
    }

    return 0;
}

static pj_status_t  auth_lookup_cred( pj_pool_t *pool, const pj_str_t *realm, const pj_str_t *acc_name, pjsip_cred_info *cred_info )
{
    pj_strdup(pool, &cred_info->realm, realm);
    pj_strdup(pool, &cred_info->username, acc_name);
    cred_info->scheme = pj_str("digest");
    cred_info->data_type = 0;
    cred_info->data = pj_str("123456");
    return PJ_SUCCESS;
}

static pj_bool_t on_rx_request( pjsip_rx_data *rdata )
{
    CGB28181DeviceMgr_ptr pGB28181DeviceMgr = GetService()->GetGB28181DeviceMgr();

    if(rdata->msg_info.msg->line.req.method.id == PJSIP_REGISTER_METHOD)
    {
        pjsip_authorization_hdr* authHdr = (pjsip_authorization_hdr*)(pjsip_msg_find_hdr(rdata->msg_info.msg, PJSIP_H_AUTHORIZATION, NULL));
        if(authHdr)
        {
            int status_code;
            if( pjsip_auth_srv_verify(&sipstk_obj_.auth_srv, rdata, &status_code) != PJ_SUCCESS )
            {
                pj_str_t reason = pj_str("Auth failed");
                pjsip_endpt_respond_stateless( sipstk_obj_.sip_endpt, rdata, status_code, &reason, NULL, NULL);
            }
            else
            {
                pGB28181DeviceMgr->OnRegister(rdata);
            }
        }
        else
        {
            sipstk_send_response(401, NULL, rdata);
        }
    }
    else if(rdata->msg_info.msg->line.req.method.id == PJSIP_INVITE_METHOD)
    {
        pGB28181DeviceMgr->OnInvite(rdata);
    }
    else if(rdata->msg_info.msg->line.req.method.id == PJSIP_ACK_METHOD)
    {
        pGB28181DeviceMgr->OnAck(rdata);
    }
    else if(rdata->msg_info.msg->line.req.method.id == PJSIP_OTHER_METHOD)
    {
        if ( pj_strcmp2(rdata->msg_info.msg->line.req.method.name, "MESSAGE") == 0 )
        {
            pGB28181DeviceMgr->OnMessage(rdata);
        }
        else if ( pj_strcmp2(rdata->msg_info.msg->line.req.method.name, "INFO") == 0 )
        {
            pGB28181DeviceMgr->OnInfo(rdata);
        }
        else
        {
            pj_str_t reason = pj_str("Unsupported Operation");
            pjsip_endpt_respond_stateless( sipstk_obj_.sip_endpt, rdata, 500, &reason, NULL, NULL);
        }
    }

    return PJ_TRUE;
}

static void call_on_media_update( pjsip_inv_session *inv, pj_status_t status)
{

}

static void call_on_state_changed( pjsip_inv_session *inv, pjsip_event *e)
{
    sip_invite_session* inv_session = inv->mod_data[sipstk_mod_.id];
    if( !inv_session || !inv_session->running_ )
    {
        return;
    }

    if (inv->state == PJSIP_INV_STATE_DISCONNECTED) 
    {
        return pGB28181DeviceMgr->OnInviteSessionDisconnected(inv_session);
    }
    else if (inv->state == PJSIP_INV_STATE_CONFIRMED)
    {

    }
    else if ( inv->state == PJSIP_INV_STATE_EARLY || inv->state == PJSIP_INV_STATE_CONNECTING )
    {

    }
}

static void call_on_forked(pjsip_inv_session *inv, pjsip_event *e)
{

}

pj_bool_t CSipStack::on_rx_request( pjsip_rx_data *rdata )
{
    pjsip_contact_hdr* contact = (pjsip_contact_hdr*)pjsip_msg_find_hdr(rdata->msg_info.msg, PJSIP_H_CONTACT, NULL);
    pjsip_sip_uri *sip_uri = (pjsip_sip_uri *)pjsip_uri_get_uri(contact->uri);

    std::string strContact = sip_uri->user.ptr;
    if(rdata->msg_info.cseq->method.id == PJSIP_REGISTER_METHOD)
    {
        //std::string strContact = rdata->msg_info.
    }
    else if(rdata->msg_info.cseq->method.id == PJSIP_INVITE_METHOD)
    {
        rdata->msg_info.require->name;
    }
    else if(rdata->msg_info.cseq->method.id == PJSIP_OTHER_METHOD)
    {

    }

    return false;
}

void CSipStack::call_on_media_update( pjsip_inv_session *inv, pj_status_t status)
{

}

void CSipStack::call_on_state_changed( pjsip_inv_session *inv, pjsip_event *e)
{

}

void CSipStack::call_on_forked(pjsip_inv_session *inv, pjsip_event *e)
{

}

