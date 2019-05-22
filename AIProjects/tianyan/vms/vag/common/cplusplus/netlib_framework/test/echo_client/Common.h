#ifndef __COMMOM_H_
#define __COMMOM_H_

#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <boost/bind.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/asio.hpp>
#include <boost/shared_array.hpp>
#include <boost/array.hpp>

typedef boost::shared_ptr<boost::asio::io_service>          IOService_ptr;
typedef boost::shared_ptr<boost::asio::ip::tcp::socket>     TCPSocket_ptr;
typedef boost::shared_ptr<boost::asio::ip::udp::socket>     UDPPSocket_ptr;
typedef boost::shared_ptr<boost::asio::io_service::work>    IOServiceWork_ptr;
typedef boost::shared_ptr<boost::asio::deadline_timer>      Timer_ptr;

#endif