/*******************************************************************************
	File:		CBitReader.h

	Contains:	the bit reader class header file

	Written by:	Bangfei Jin

	Change History (most recent first):
	2016-12-08		Bangfei			Create file

*******************************************************************************/
#ifndef __CBitReader_H__
#define __CBitReader_H__


class CBitReader
{
public:
	CBitReader(unsigned char *pData, unsigned int nSize);
    virtual ~CBitReader();

    unsigned int	GetBits (unsigned int n);
    void			SkipBits (unsigned int n);

	void			PutBits (unsigned int x, unsigned int n);

    unsigned int	numBitsLeft (void) const;

    unsigned char *	Data() const;

protected:
    unsigned char * m_pData;
    unsigned int	m_nSize;

    unsigned int	m_nReservoir;	// left-aligned bits
    unsigned int	m_nNumBitsLeft;

    virtual void	FillReservoir(void);
};

#endif  // __CBitReader_H__
