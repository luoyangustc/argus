package util

import (
	"errors"
	"math/rand"
	"net"
	"strconv"
	"time"
)

func CheckPort(port string) (free bool) {
	conn, err := net.DialTimeout("tcp", net.JoinHostPort("", port), 1*time.Second)
	if conn != nil {
		conn.Close()
		free = false
	}
	if err != nil || conn == nil {
		free = true
	}
	return
}

func allocPort() (port string) {
	i_port := 8000 + rand.Intn(999)
	port = strconv.Itoa(i_port)
	return
}

func AllocPort() (port string, err error) {
	for i := 0; i < 5; i++ {
		port = allocPort()
		if CheckPort(port) {
			return
		}
	}
	err = errors.New("alloc port Fail!")
	return
}

func Localhost() (ip string, err error) {
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return 
	}
	for _, address := range addrs {
		// 检查ip地址判断是否回环地址
		if ipnet, ok := address.(*net.IPNet); ok && !ipnet.IP.IsLoopback() {
			if ipnet.IP.To4() != nil {
				//fmt.Println(ipnet.IP.String())
				ip = ipnet.IP.String()
				return
			}

		}
	}
	return
}
