package dht

type NodeInfo struct {
	Host string
	Key  []byte
}

type NodeInfos []NodeInfo

type RouterInfo struct {
	Host    string
	Metrics int
}

type RouterInfos []RouterInfo

type Interface interface {
	Setup(nodes NodeInfos)
	Nodes() NodeInfos
	Route(key []byte, ttl int) (routers RouterInfos) // len(routes) <= ttl
}
