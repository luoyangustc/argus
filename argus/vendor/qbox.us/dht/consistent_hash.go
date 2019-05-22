package dht

type ConsistentHash struct {
	nodes NodeInfos
}

func NewConsistentHash(nodes NodeInfos) *ConsistentHash {
	ch := &ConsistentHash{}
	ch.Setup(nodes)
	return ch
}

func (p *ConsistentHash) Setup(nodes NodeInfos) {
	p.nodes = nodes
}

func (p *ConsistentHash) Nodes() NodeInfos {
	return p.nodes
}

func (p *ConsistentHash) Route(key []byte, ttl int) (routers RouterInfos) {
	return []RouterInfo{{Host: p.nodes[0].Host, Metrics: 1}}
}
