package middleware

import (
	"context"
	"reflect"

	"github.com/go-kit/kit/endpoint"
)

// type ServiceA interface {
// 	FooA(_ context.Context, req string) (resp int, err error)
// 	FooB(_ context.Context, req struct{}) (resp struct{}, err error)
// }
//
// type serviceA struct{}
//
// func (s serviceA) FooA(_ context.Context, req string) (resp int, err error)        { return }
// func (s serviceA) FooB(_ context.Context, req struct{}) (resp struct{}, err error) { return }
//
// type ServiceAEndpoints struct {
// 	FooAEP endpoint.Endpoint
// 	FooBEP endpoint.Endpoint
// }
//
// func (e ServiceAEndpoints) FooA(_ context.Context, req string) (resp int, err error)        { return }
// func (e ServiceAEndpoints) FooB(_ context.Context, req struct{}) (resp struct{}, err error) { return }

type Service interface{}
type ServiceEndpoints interface{}

type EndpointFactory func(methodName string, service Service, defaultEndpoint func() endpoint.Endpoint) (endpoint.Endpoint, bool)
type DefaultEndpointFactory func(methodName string, service Service, defaultEndpoint func() endpoint.Endpoint) endpoint.Endpoint

var defaultEndpoint = func(sv reflect.Value, mn string) func() endpoint.Endpoint {
	return func() endpoint.Endpoint {
		return func(ctx context.Context, request interface{}) (response interface{}, err error) {
			rets := sv.MethodByName(mn).Call([]reflect.Value{reflect.ValueOf(ctx), reflect.ValueOf(request)})
			if rets[0].IsValid() {
				response = rets[0].Interface()
			}
			if rets[1].IsValid() && !rets[1].IsNil() {
				err = rets[1].Interface().(error)
			}
			return
		}
	}
}

func MakeMiddlewareFactory(
	factory EndpointFactory, defaultFactory DefaultEndpointFactory,
) func(service Service, endpoints ServiceEndpoints) (Service, error) {
	return func(service Service, endpoints ServiceEndpoints) (Service, error) {
		return MakeMiddleware(service, endpoints, factory, defaultFactory)
	}
}

func MakeMiddleware(
	service Service, endpoints ServiceEndpoints,
	factory EndpointFactory, defaultFactory DefaultEndpointFactory,
) (Service, error) {

	sv := reflect.ValueOf(service)
	st := sv.Type()

	ev := reflect.ValueOf(endpoints)
	if !ev.CanSet() {
		ev = reflect.New(ev.Type())
	}

	for i, n := 0, st.NumMethod(); i < n; i++ {
		mn := st.Method(i).Name

		// TODO 支持flag
		fv := ev.Elem().FieldByName(mn + "EP")
		if !fv.IsValid() || !fv.CanSet() {
			continue
		}
		dep := defaultEndpoint(sv, mn)

		if factory != nil {
			ep, ok := factory(mn, service, dep)
			if ok {
				fv.Set(reflect.ValueOf(ep))
				continue
			}
		}
		if defaultFactory != nil {
			edv := defaultFactory(mn, service, dep)
			fv.Set(reflect.ValueOf(edv))
		} else {
			fv.Set(reflect.ValueOf(dep()))
		}
	}

	return ev.Elem().Interface(), nil
}

////////////////////////////////////////////////////////////////////////////////

type ServiceFactory struct {
	service  Service
	New      func() Service
	NewShell func() ServiceEndpoints
}

func (f *ServiceFactory) Gen() Service {
	if f.service == nil {
		f.service = f.New()
	}
	return f.service
}

type ServiceChain struct {
	layers []ServiceLayer
	*ServiceFactory
}

func NewServiceChain(sf *ServiceFactory) *ServiceChain {
	return &ServiceChain{layers: make([]ServiceLayer, 0), ServiceFactory: sf}
}

type ServiceLayer interface {
	New(Service, ServiceEndpoints) (Service, error)
}

func (chain *ServiceChain) AddLayer(layer ServiceLayer) *ServiceChain {
	chain.layers = append(chain.layers, layer)
	return chain
}

func (chain *ServiceChain) Gen() (Service, error) {
	s := chain.ServiceFactory.Gen()
	for _, layer := range chain.layers {
		ss, err := layer.New(s, chain.ServiceFactory.NewShell())
		if err != nil {
			return nil, err
		}
		s = ss
	}
	return s, nil
}

func (chain *ServiceChain) Kernel() Service { return chain.ServiceFactory.Gen() }
