/*******************************************************************************
	File:		CBitReader.cpp

	Contains:	the bit reader class implement file

	Written by:	Bangfei Jin

	Change History (most recent first):
	2016-12-08		Bangfei			Create file

*******************************************************************************/

#include "CBitReader.h"

CBitReader::CBitReader(unsigned char *pData, unsigned int nSize)
	: m_pData(pData)
    , m_nSize(nSize)
    , m_nReservoir(0)
    , m_nNumBitsLeft(0) 
{
}

CBitReader::~CBitReader() 
{
}

void CBitReader::FillReservoir (void)
{
    m_nReservoir = 0;
    unsigned int  i;
    for (i = 0; m_nSize > 0 && i < 4; ++i)
	{
        m_nReservoir = (m_nReservoir << 8) | *m_pData;

        ++m_pData;
        --m_nSize;
    }

    m_nNumBitsLeft = 8 * i;
    m_nReservoir <<= 32 - m_nNumBitsLeft;
}

unsigned int  CBitReader::GetBits (unsigned int n)
{
    unsigned int  result = 0;
    while (n > 0)
	{
        if (m_nNumBitsLeft == 0) 
            FillReservoir();

		if (numBitsLeft() == 0)
		{
			break;
		}
        unsigned int  m = n;
        if (m > m_nNumBitsLeft) 
            m = m_nNumBitsLeft;

        result = (result << m) | (m_nReservoir >> (32 - m));
        m_nReservoir <<= m;
        m_nNumBitsLeft -= m;

        n -= m;
    }

    return result;
}

void CBitReader::SkipBits (unsigned int  n) 
{
    while (n > 32) 
	{
        GetBits(32);
        n -= 32;
    }

    if (n > 0) 
        GetBits(n);
}

void CBitReader::PutBits (unsigned int  x, unsigned int  n) 
{
    while (m_nNumBitsLeft + n > 32)
	{
        m_nNumBitsLeft -= 8;
        --m_pData;
        ++m_nSize;
    }

    m_nReservoir = (m_nReservoir >> n) | (x << (32 - n));
    m_nNumBitsLeft += n;
}

unsigned int  CBitReader::numBitsLeft (void) const 
{
    return m_nSize * 8 + m_nNumBitsLeft;
}

unsigned char *CBitReader::Data (void) const 
{
    return m_pData - (m_nNumBitsLeft + 7) / 8;
}

