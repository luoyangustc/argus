#include "LBitField.h"
#include <string.h>
CLBitField::CLBitField()
{	
	m_iDataLen = 0;
	m_iBitCount = 0;
	m_iHasBitCount = 0;
	memset(m_pData,0,BITMAP_BYTE_CNT);
}

CLBitField::~CLBitField()
{}
//���ñ�����ĳߴ�
void CLBitField::SetFieldSize(int iBitCount)
{
	m_iDataLen = (iBitCount/8)+((iBitCount%8)?1:0);
	m_iBitCount = iBitCount;
	m_iHasBitCount = 0;
}

//�õ�ָ��������ݵ�ֵ,����TRUE,��ʾָ���������Ѿ���д.
BOOL CLBitField::GetBitValue(DWORD dwBit) const
{
	/*! PGP, 2009-8-4   12:01
	*	��Ϊ��m_iBitCountת����DWORD����Ȼ������dwBitΪ��1ʱ���͹���
	*/
	if(dwBit>=static_cast<DWORD>(m_iBitCount)||m_pData==NULL)
		return FALSE;				
	return (m_pData[dwBit/8]&(1<<(7-(dwBit%8))))?TRUE:FALSE;
}
//���óɹ�����TRUE
BOOL CLBitField::SetBitValue(DWORD dwBit,BOOL bValue)
{
	if(static_cast<int>(dwBit)>=m_iBitCount||m_pData==NULL)
		return FALSE;	
	BOOL bOldValue = (m_pData[dwBit/8]&(1<<(7-(dwBit%8))))?TRUE:FALSE;
	if(bValue)
	{
		if(!bOldValue)
		{
			if(m_iHasBitCount<m_iBitCount)
				m_iHasBitCount++;
		}
		m_pData[dwBit/8] |= 1<<(7-(dwBit%8));
	}
	else
	{
		if(bOldValue)
		{			
			if(m_iHasBitCount>0)
				m_iHasBitCount--;				
		}
		m_pData[dwBit/8] &= ~(1<<(7-(dwBit%8)));
	}
	return TRUE;
	//return bOldValue;
}
//��ʼ��������,bValueΪTRUE,ȫ����ʼ��Ϊ1,����ȫ����ʼ��Ϊ0
void CLBitField::init(BOOL bValue)
{
	if(bValue)
	{
		memset(m_pData,0xFFFFFFFF,m_iDataLen);
		int iBitCount = m_iBitCount;
		while(iBitCount%8)
		{
			m_pData[iBitCount/8] &= ~(1<<(7-(iBitCount%8)));
			iBitCount++;
		}
		m_iHasBitCount = m_iBitCount;
	}
	else
	{
		memset(m_pData,0,m_iDataLen);
		m_iHasBitCount = 0;
	}
}

//������Ϣ���ݳ�ʼ��������ṹ
void CLBitField::initbymsg(BYTE *pMsgData)
{
	if(!pMsgData)
		return ;
	memcpy(m_pData,pMsgData,m_iDataLen);
	//ͳ�ưٷֱ�
	m_iHasBitCount = 0;
	for(int i = 0;i<m_iBitCount;i++)
	{
		if(GetBitValue((DWORD)i))
			m_iHasBitCount++;
	}
}

//�õ����ؽ���,
float CLBitField::GetPercent() const
{
	float f1 = (float)m_iBitCount;
	float f2 = (float)m_iHasBitCount;
	return (0==m_iBitCount || m_iHasBitCount==0)?((float)0.0):(f2*100)/f1;
}


CLBitField & CLBitField::operator=(const CLBitField& _Right)
{
	memcpy(this, &_Right, sizeof(CLBitField));
	return * this;
}

bool CLBitField::operator == (const CLBitField& _Right) const
{
	if( m_iBitCount == _Right.m_iBitCount
		&& m_iHasBitCount == _Right.m_iHasBitCount
		&& m_iDataLen == _Right.m_iDataLen
		&& 0 == memcmp(m_pData, _Right.m_pData, m_iDataLen))
		return true;
	else
		return false;
}

