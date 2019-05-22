# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensord.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensord.proto',
  package='tensord.grpc',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\rtensord.proto\x12\x0ctensord.grpc\"\"\n\x04\x44\x61ta\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0c\n\x04\x62ody\x18\x02 \x01(\x0c\"\x88\x01\n\x08Requests\x12\r\n\x05model\x18\x01 \x01(\t\x12\x0f\n\x07version\x18\x02 \x01(\x05\x12/\n\x07request\x18\x03 \x03(\x0b\x32\x1e.tensord.grpc.Requests.Request\x1a+\n\x07Request\x12 \n\x04\x64\x61ta\x18\x01 \x03(\x0b\x32\x12.tensord.grpc.Data\"m\n\tResponses\x12\x32\n\x08response\x18\x03 \x03(\x0b\x32 .tensord.grpc.Responses.Response\x1a,\n\x08Response\x12 \n\x04\x64\x61ta\x18\x01 \x03(\x0b\x32\x12.tensord.grpc.Data2G\n\x07Tensord\x12<\n\x07Predict\x12\x16.tensord.grpc.Requests\x1a\x17.tensord.grpc.Responses\"\x00\x62\x06proto3')
)




_DATA = _descriptor.Descriptor(
  name='Data',
  full_name='tensord.grpc.Data',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='tensord.grpc.Data.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='body', full_name='tensord.grpc.Data.body', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=31,
  serialized_end=65,
)


_REQUESTS_REQUEST = _descriptor.Descriptor(
  name='Request',
  full_name='tensord.grpc.Requests.Request',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='tensord.grpc.Requests.Request.data', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=161,
  serialized_end=204,
)

_REQUESTS = _descriptor.Descriptor(
  name='Requests',
  full_name='tensord.grpc.Requests',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='model', full_name='tensord.grpc.Requests.model', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='version', full_name='tensord.grpc.Requests.version', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='request', full_name='tensord.grpc.Requests.request', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_REQUESTS_REQUEST, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=68,
  serialized_end=204,
)


_RESPONSES_RESPONSE = _descriptor.Descriptor(
  name='Response',
  full_name='tensord.grpc.Responses.Response',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='tensord.grpc.Responses.Response.data', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=271,
  serialized_end=315,
)

_RESPONSES = _descriptor.Descriptor(
  name='Responses',
  full_name='tensord.grpc.Responses',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='response', full_name='tensord.grpc.Responses.response', index=0,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_RESPONSES_RESPONSE, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=206,
  serialized_end=315,
)

_REQUESTS_REQUEST.fields_by_name['data'].message_type = _DATA
_REQUESTS_REQUEST.containing_type = _REQUESTS
_REQUESTS.fields_by_name['request'].message_type = _REQUESTS_REQUEST
_RESPONSES_RESPONSE.fields_by_name['data'].message_type = _DATA
_RESPONSES_RESPONSE.containing_type = _RESPONSES
_RESPONSES.fields_by_name['response'].message_type = _RESPONSES_RESPONSE
DESCRIPTOR.message_types_by_name['Data'] = _DATA
DESCRIPTOR.message_types_by_name['Requests'] = _REQUESTS
DESCRIPTOR.message_types_by_name['Responses'] = _RESPONSES
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Data = _reflection.GeneratedProtocolMessageType('Data', (_message.Message,), dict(
  DESCRIPTOR = _DATA,
  __module__ = 'tensord_pb2'
  # @@protoc_insertion_point(class_scope:tensord.grpc.Data)
  ))
_sym_db.RegisterMessage(Data)

Requests = _reflection.GeneratedProtocolMessageType('Requests', (_message.Message,), dict(

  Request = _reflection.GeneratedProtocolMessageType('Request', (_message.Message,), dict(
    DESCRIPTOR = _REQUESTS_REQUEST,
    __module__ = 'tensord_pb2'
    # @@protoc_insertion_point(class_scope:tensord.grpc.Requests.Request)
    ))
  ,
  DESCRIPTOR = _REQUESTS,
  __module__ = 'tensord_pb2'
  # @@protoc_insertion_point(class_scope:tensord.grpc.Requests)
  ))
_sym_db.RegisterMessage(Requests)
_sym_db.RegisterMessage(Requests.Request)

Responses = _reflection.GeneratedProtocolMessageType('Responses', (_message.Message,), dict(

  Response = _reflection.GeneratedProtocolMessageType('Response', (_message.Message,), dict(
    DESCRIPTOR = _RESPONSES_RESPONSE,
    __module__ = 'tensord_pb2'
    # @@protoc_insertion_point(class_scope:tensord.grpc.Responses.Response)
    ))
  ,
  DESCRIPTOR = _RESPONSES,
  __module__ = 'tensord_pb2'
  # @@protoc_insertion_point(class_scope:tensord.grpc.Responses)
  ))
_sym_db.RegisterMessage(Responses)
_sym_db.RegisterMessage(Responses.Response)



_TENSORD = _descriptor.ServiceDescriptor(
  name='Tensord',
  full_name='tensord.grpc.Tensord',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=317,
  serialized_end=388,
  methods=[
  _descriptor.MethodDescriptor(
    name='Predict',
    full_name='tensord.grpc.Tensord.Predict',
    index=0,
    containing_service=None,
    input_type=_REQUESTS,
    output_type=_RESPONSES,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_TENSORD)

DESCRIPTOR.services_by_name['Tensord'] = _TENSORD

# @@protoc_insertion_point(module_scope)