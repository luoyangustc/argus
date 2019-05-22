#include "protocol_exchangekey.h"
namespace protocol{
CDataStream& operator<<( CDataStream& _ds,ExchangeKeyRequest & _msgdata )
{
	_ds << _msgdata.mask;
	if( 0x01 & _msgdata.mask )
	{
		_ds << _msgdata.key_P_length;
		if( _msgdata.key_P_length && _msgdata.key_P_length <= sizeof(_msgdata.key_P) )
		{
			_ds.writedata( _msgdata.key_P,_msgdata.key_P_length);
		}
		
		_ds << _msgdata.key_A_length;
		if( _msgdata.key_A_length && _msgdata.key_A_length <= sizeof(_msgdata.key_A) )
		{
			_ds.writedata( _msgdata.key_A,_msgdata.key_A_length);
		}
	}

	if( 0x02 & _msgdata.mask )
	{
		_ds << _msgdata.except_algorithm;
        _ds << _msgdata.algorithm_param;
	}

	return _ds;
}

CDataStream& operator>>( CDataStream& _ds, ExchangeKeyRequest& _msgdata )
{
	_ds >> _msgdata.mask;
	if( 0x01 & _msgdata.mask )
	{
		_ds >> _msgdata.key_P_length;
		if( _msgdata.key_P_length && _msgdata.key_P_length <= sizeof(_msgdata.key_P) )
		{
			_ds.readdata( _msgdata.key_P_length,_msgdata.key_P);
		}
		_ds >> _msgdata.key_A_length;
		if( _msgdata.key_A_length && _msgdata.key_A_length <= sizeof(_msgdata.key_A) )
		{
			_ds.readdata(_msgdata.key_A_length, _msgdata.key_A);
		}
	}

	if( 0x02 & _msgdata.mask )
	{
		_ds >> _msgdata.except_algorithm;
        _ds >> _msgdata.algorithm_param;
	}
	return _ds;
}

CDataStream& operator<<( CDataStream& _ds,ExchangeKeyResponse & _msgdata )
{
	_ds << _msgdata.mask;
    _ds << _msgdata.resp_code;

	if( 0x01 & _msgdata.mask )
	{
		_ds << _msgdata.key_B_length;
		if( _msgdata.key_B_length && _msgdata.key_B_length <= sizeof(_msgdata.key_B) )
		{
			_ds.writedata( _msgdata.key_B,_msgdata.key_B_length);
		}
		_ds << _msgdata.key_size;
	}

	if( 0x02 & _msgdata.mask )
	{
		_ds << _msgdata.encry_algorithm;
        _ds << _msgdata.algorithm_param;
	}
	return _ds;
}

CDataStream& operator>>( CDataStream& _ds, ExchangeKeyResponse& _msgdata )
{
	_ds >> _msgdata.mask;
    _ds >> _msgdata.resp_code;
	if( 0x01 & _msgdata.mask )
	{
		_ds >> _msgdata.key_B_length;
		if( _msgdata.key_B_length && _msgdata.key_B_length <= sizeof(_msgdata.key_B) )
		{
			_ds.readdata(_msgdata.key_B_length, _msgdata.key_B);
		}
        else
        {
        }
		_ds >> _msgdata.key_size;
	}

	if( 0x02 & _msgdata.mask )
	{
		_ds >> _msgdata.encry_algorithm;
        _ds >> _msgdata.algorithm_param;
	}
	return _ds;
}
}

