#ifndef __L_Bit_Field_H__
#define __L_Bit_Field_H__

#include "typedef_win.h"
#include <string>
using namespace std;

#define BITMAP_BYTE_CNT 32
#define BITMAP_BYTE_CNTEX BITMAP_BYTE_CNT*4

class CLBitField  
{
public:
	float GetPercent() const;
	void initbymsg(BYTE * pMsgData);
	void init(BOOL bValue);
	bool IsInit(void){return m_iBitCount !=0;}
	BOOL GetBitValue(DWORD dwBit) const;
	BOOL SetBitValue(DWORD dwBit,BOOL bValue);
	void SetFieldSize(int iBitCount);
	bool IsFull(void){return (m_iBitCount>0 && m_iBitCount == m_iHasBitCount);}
	
	CLBitField();
	virtual ~CLBitField();
	BYTE m_pData[BITMAP_BYTE_CNT];
	int m_iDataLen;	
	int m_iBitCount;
	int m_iHasBitCount;

	CLBitField & operator=(const CLBitField& _Right);
	bool operator == (const CLBitField& _Right) const;
};

#endif // __L_Bit_Field_H__
