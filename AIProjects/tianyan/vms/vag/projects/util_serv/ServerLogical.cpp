#include "ServerLogical.h"

// FIXME : ?
CServerLogical CServerLogical::logic_;

CServerLogical* GetService()
{
  return CServerLogical::GetLogical();
}

IServerLogical * IServerLogical::GetInstance()
{
	return CServerLogical::GetLogical();
}

CServerLogical::CServerLogical()
{
}

CServerLogical::~CServerLogical()
{
}

bool CServerLogical::Start(uint16 http_port, uint16 serv_port)
{
	start_tick_ = get_current_tick();

  pSysMonitorThread_.reset(new CSysMonitorThread);
  if (pSysMonitorThread_)
  {
    pSysMonitorThread_->Start();
  }

	return true;
}

void CServerLogical::Stop()
{
}

void CServerLogical::Update()
{
  pSysMonitorThread_->UpdateActiveTick();   
}

void CServerLogical::DoIdleTask()
{
}

int32 CServerLogical::OnTCPAccepted(ITCPSessionSendSink*sink,
    CHostInfo& hiRemote,
    CDataStream& sendds)
{
  Info("message(%x,%u) accepted !", hiRemote.IP, hiRemote.Port);	
  return 0;
}

int32 CServerLogical::OnTCPClosed(ITCPSessionSendSink*sink, CHostInfo& hiRemote)
{
  Info("message(%x,%u) closed !", hiRemote.IP, hiRemote.Port);
  return 0;
}
