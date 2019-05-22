package middleware

import (
	"context"
	"errors"
	"testing"

	"github.com/go-kit/kit/endpoint"
	"github.com/stretchr/testify/assert"
)

type IServiceA interface {
	Foo(_ context.Context, request string) (response string, err error)
}

type ServiceA struct{}

func (s ServiceA) Foo(_ context.Context, request string) (response string, err error) {
	return "", errors.New(request)
}

type ServiceAEndpoints struct {
	FooEP endpoint.Endpoint
}

func (ep ServiceAEndpoints) Foo(ctx context.Context, request string) (response string, err error) {
	_, err = ep.FooEP(ctx, request)
	return "", err
}

func TestFoo(t *testing.T) {

	s, _ := MakeMiddleware(ServiceA{}, ServiceAEndpoints{}, nil, nil)
	_, err := s.(IServiceA).Foo(context.Background(), "xxx")
	assert.Equal(t, "xxx", err.Error())

}
