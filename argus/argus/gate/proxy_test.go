package gate

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestProxyRoutes(t *testing.T) {

	rs := NewProxyRoutes()

	{
		_, ok := rs.Get("/a/b")
		assert.False(t, ok)
	}
	{
		rs.Set("/a/b/",
			ProxyRoute{
				Path: "/a/b/",
				Host: "host1",
			})
		route, ok := rs.Get("/a/b/")
		assert.True(t, ok)
		assert.Equal(t, "host1", route.Host)
	}
	{
		rs.Set("/a/b",
			ProxyRoute{
				Path: "/a/b",
				Host: "host2",
			})
		route, ok := rs.Get("/a/b")
		assert.True(t, ok)
		assert.Equal(t, "host2", route.Host)
		route, ok = rs.Get("/a/b/")
		assert.True(t, ok)
		assert.Equal(t, "host1", route.Host)
		_, ok = rs.Get("/a/c")
		assert.False(t, ok)
	}
	{
		rs.Del("/a/b/")
		route, ok := rs.Get("/a/b")
		assert.True(t, ok)
		assert.Equal(t, "host2", route.Host)
		_, ok = rs.Get("/a/b/")
		assert.False(t, ok)
		_, ok = rs.Get("/a/c")
		assert.False(t, ok)
	}
	{
		rs.Set("/a/b/c/d", ProxyRoute{Path: "/a/b/c/d", Host: "host4"})
		rs.Set("/a/b/b", ProxyRoute{Path: "/a/b/b", Host: "host2"})
		rs.Set("/a/b/c", ProxyRoute{Path: "/a/b/c", Host: "host3"})
		route, ok := rs.Get("/a/b/b")
		assert.True(t, ok)
		assert.Equal(t, "host2", route.Host)
		route, ok = rs.Get("/a/b/c")
		assert.True(t, ok)
		assert.Equal(t, "host3", route.Host)
		route, ok = rs.Get("/a/b/c/d")
		assert.True(t, ok)
		assert.Equal(t, "host4", route.Host)
	}

}
