// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: forward.proto

#ifndef PROTOBUF_INCLUDED_forward_2eproto
#define PROTOBUF_INCLUDED_forward_2eproto

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3006001
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3006001 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/inlined_string_field.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#define PROTOBUF_INTERNAL_EXPORT_protobuf_forward_2eproto 

namespace protobuf_forward_2eproto {
// Internal implementation detail -- do not use these members.
struct TableStruct {
  static const ::google::protobuf::internal::ParseTableField entries[];
  static const ::google::protobuf::internal::AuxillaryParseTableField aux[];
  static const ::google::protobuf::internal::ParseTable schema[5];
  static const ::google::protobuf::internal::FieldMetadata field_metadata[];
  static const ::google::protobuf::internal::SerializationTable serialization_table[];
  static const ::google::protobuf::uint32 offsets[];
};
void AddDescriptors();
}  // namespace protobuf_forward_2eproto
namespace inference {
namespace wa {
class ForwardRequest;
class ForwardRequestDefaultTypeInternal;
extern ForwardRequestDefaultTypeInternal _ForwardRequest_default_instance_;
class ForwardRequest_Data;
class ForwardRequest_DataDefaultTypeInternal;
extern ForwardRequest_DataDefaultTypeInternal _ForwardRequest_Data_default_instance_;
class ForwardResponse;
class ForwardResponseDefaultTypeInternal;
extern ForwardResponseDefaultTypeInternal _ForwardResponse_default_instance_;
class ForwardResponse_Box;
class ForwardResponse_BoxDefaultTypeInternal;
extern ForwardResponse_BoxDefaultTypeInternal _ForwardResponse_Box_default_instance_;
class ForwardResponse_Label;
class ForwardResponse_LabelDefaultTypeInternal;
extern ForwardResponse_LabelDefaultTypeInternal _ForwardResponse_Label_default_instance_;
}  // namespace wa
}  // namespace inference
namespace google {
namespace protobuf {
template<> ::inference::wa::ForwardRequest* Arena::CreateMaybeMessage<::inference::wa::ForwardRequest>(Arena*);
template<> ::inference::wa::ForwardRequest_Data* Arena::CreateMaybeMessage<::inference::wa::ForwardRequest_Data>(Arena*);
template<> ::inference::wa::ForwardResponse* Arena::CreateMaybeMessage<::inference::wa::ForwardResponse>(Arena*);
template<> ::inference::wa::ForwardResponse_Box* Arena::CreateMaybeMessage<::inference::wa::ForwardResponse_Box>(Arena*);
template<> ::inference::wa::ForwardResponse_Label* Arena::CreateMaybeMessage<::inference::wa::ForwardResponse_Label>(Arena*);
}  // namespace protobuf
}  // namespace google
namespace inference {
namespace wa {

// ===================================================================

class ForwardRequest_Data : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:inference.wa.ForwardRequest.Data) */ {
 public:
  ForwardRequest_Data();
  virtual ~ForwardRequest_Data();

  ForwardRequest_Data(const ForwardRequest_Data& from);

  inline ForwardRequest_Data& operator=(const ForwardRequest_Data& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  ForwardRequest_Data(ForwardRequest_Data&& from) noexcept
    : ForwardRequest_Data() {
    *this = ::std::move(from);
  }

  inline ForwardRequest_Data& operator=(ForwardRequest_Data&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor();
  static const ForwardRequest_Data& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const ForwardRequest_Data* internal_default_instance() {
    return reinterpret_cast<const ForwardRequest_Data*>(
               &_ForwardRequest_Data_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  void Swap(ForwardRequest_Data* other);
  friend void swap(ForwardRequest_Data& a, ForwardRequest_Data& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline ForwardRequest_Data* New() const final {
    return CreateMaybeMessage<ForwardRequest_Data>(NULL);
  }

  ForwardRequest_Data* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<ForwardRequest_Data>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const ForwardRequest_Data& from);
  void MergeFrom(const ForwardRequest_Data& from);
  void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) final;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const final;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(ForwardRequest_Data* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // bytes body = 3;
  void clear_body();
  static const int kBodyFieldNumber = 3;
  const ::std::string& body() const;
  void set_body(const ::std::string& value);
  #if LANG_CXX11
  void set_body(::std::string&& value);
  #endif
  void set_body(const char* value);
  void set_body(const void* value, size_t size);
  ::std::string* mutable_body();
  ::std::string* release_body();
  void set_allocated_body(::std::string* body);

  // @@protoc_insertion_point(class_scope:inference.wa.ForwardRequest.Data)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::internal::ArenaStringPtr body_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::protobuf_forward_2eproto::TableStruct;
};
// -------------------------------------------------------------------

class ForwardRequest : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:inference.wa.ForwardRequest) */ {
 public:
  ForwardRequest();
  virtual ~ForwardRequest();

  ForwardRequest(const ForwardRequest& from);

  inline ForwardRequest& operator=(const ForwardRequest& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  ForwardRequest(ForwardRequest&& from) noexcept
    : ForwardRequest() {
    *this = ::std::move(from);
  }

  inline ForwardRequest& operator=(ForwardRequest&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor();
  static const ForwardRequest& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const ForwardRequest* internal_default_instance() {
    return reinterpret_cast<const ForwardRequest*>(
               &_ForwardRequest_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  void Swap(ForwardRequest* other);
  friend void swap(ForwardRequest& a, ForwardRequest& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline ForwardRequest* New() const final {
    return CreateMaybeMessage<ForwardRequest>(NULL);
  }

  ForwardRequest* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<ForwardRequest>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const ForwardRequest& from);
  void MergeFrom(const ForwardRequest& from);
  void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) final;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const final;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(ForwardRequest* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  typedef ForwardRequest_Data Data;

  // accessors -------------------------------------------------------

  // .inference.wa.ForwardRequest.Data data = 1;
  bool has_data() const;
  void clear_data();
  static const int kDataFieldNumber = 1;
  private:
  const ::inference::wa::ForwardRequest_Data& _internal_data() const;
  public:
  const ::inference::wa::ForwardRequest_Data& data() const;
  ::inference::wa::ForwardRequest_Data* release_data();
  ::inference::wa::ForwardRequest_Data* mutable_data();
  void set_allocated_data(::inference::wa::ForwardRequest_Data* data);

  // int32 h = 5;
  void clear_h();
  static const int kHFieldNumber = 5;
  ::google::protobuf::int32 h() const;
  void set_h(::google::protobuf::int32 value);

  // int32 w = 6;
  void clear_w();
  static const int kWFieldNumber = 6;
  ::google::protobuf::int32 w() const;
  void set_w(::google::protobuf::int32 value);

  // @@protoc_insertion_point(class_scope:inference.wa.ForwardRequest)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::inference::wa::ForwardRequest_Data* data_;
  ::google::protobuf::int32 h_;
  ::google::protobuf::int32 w_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::protobuf_forward_2eproto::TableStruct;
};
// -------------------------------------------------------------------

class ForwardResponse_Box : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:inference.wa.ForwardResponse.Box) */ {
 public:
  ForwardResponse_Box();
  virtual ~ForwardResponse_Box();

  ForwardResponse_Box(const ForwardResponse_Box& from);

  inline ForwardResponse_Box& operator=(const ForwardResponse_Box& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  ForwardResponse_Box(ForwardResponse_Box&& from) noexcept
    : ForwardResponse_Box() {
    *this = ::std::move(from);
  }

  inline ForwardResponse_Box& operator=(ForwardResponse_Box&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor();
  static const ForwardResponse_Box& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const ForwardResponse_Box* internal_default_instance() {
    return reinterpret_cast<const ForwardResponse_Box*>(
               &_ForwardResponse_Box_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    2;

  void Swap(ForwardResponse_Box* other);
  friend void swap(ForwardResponse_Box& a, ForwardResponse_Box& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline ForwardResponse_Box* New() const final {
    return CreateMaybeMessage<ForwardResponse_Box>(NULL);
  }

  ForwardResponse_Box* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<ForwardResponse_Box>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const ForwardResponse_Box& from);
  void MergeFrom(const ForwardResponse_Box& from);
  void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) final;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const final;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(ForwardResponse_Box* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // float xmin = 1;
  void clear_xmin();
  static const int kXminFieldNumber = 1;
  float xmin() const;
  void set_xmin(float value);

  // float ymin = 2;
  void clear_ymin();
  static const int kYminFieldNumber = 2;
  float ymin() const;
  void set_ymin(float value);

  // float xmax = 3;
  void clear_xmax();
  static const int kXmaxFieldNumber = 3;
  float xmax() const;
  void set_xmax(float value);

  // float ymax = 4;
  void clear_ymax();
  static const int kYmaxFieldNumber = 4;
  float ymax() const;
  void set_ymax(float value);

  // float score = 5;
  void clear_score();
  static const int kScoreFieldNumber = 5;
  float score() const;
  void set_score(float value);

  // int32 label = 6;
  void clear_label();
  static const int kLabelFieldNumber = 6;
  ::google::protobuf::int32 label() const;
  void set_label(::google::protobuf::int32 value);

  // @@protoc_insertion_point(class_scope:inference.wa.ForwardResponse.Box)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  float xmin_;
  float ymin_;
  float xmax_;
  float ymax_;
  float score_;
  ::google::protobuf::int32 label_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::protobuf_forward_2eproto::TableStruct;
};
// -------------------------------------------------------------------

class ForwardResponse_Label : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:inference.wa.ForwardResponse.Label) */ {
 public:
  ForwardResponse_Label();
  virtual ~ForwardResponse_Label();

  ForwardResponse_Label(const ForwardResponse_Label& from);

  inline ForwardResponse_Label& operator=(const ForwardResponse_Label& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  ForwardResponse_Label(ForwardResponse_Label&& from) noexcept
    : ForwardResponse_Label() {
    *this = ::std::move(from);
  }

  inline ForwardResponse_Label& operator=(ForwardResponse_Label&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor();
  static const ForwardResponse_Label& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const ForwardResponse_Label* internal_default_instance() {
    return reinterpret_cast<const ForwardResponse_Label*>(
               &_ForwardResponse_Label_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    3;

  void Swap(ForwardResponse_Label* other);
  friend void swap(ForwardResponse_Label& a, ForwardResponse_Label& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline ForwardResponse_Label* New() const final {
    return CreateMaybeMessage<ForwardResponse_Label>(NULL);
  }

  ForwardResponse_Label* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<ForwardResponse_Label>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const ForwardResponse_Label& from);
  void MergeFrom(const ForwardResponse_Label& from);
  void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) final;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const final;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(ForwardResponse_Label* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // int32 index = 1;
  void clear_index();
  static const int kIndexFieldNumber = 1;
  ::google::protobuf::int32 index() const;
  void set_index(::google::protobuf::int32 value);

  // float score = 2;
  void clear_score();
  static const int kScoreFieldNumber = 2;
  float score() const;
  void set_score(float value);

  // @@protoc_insertion_point(class_scope:inference.wa.ForwardResponse.Label)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::int32 index_;
  float score_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::protobuf_forward_2eproto::TableStruct;
};
// -------------------------------------------------------------------

class ForwardResponse : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:inference.wa.ForwardResponse) */ {
 public:
  ForwardResponse();
  virtual ~ForwardResponse();

  ForwardResponse(const ForwardResponse& from);

  inline ForwardResponse& operator=(const ForwardResponse& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  ForwardResponse(ForwardResponse&& from) noexcept
    : ForwardResponse() {
    *this = ::std::move(from);
  }

  inline ForwardResponse& operator=(ForwardResponse&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor();
  static const ForwardResponse& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const ForwardResponse* internal_default_instance() {
    return reinterpret_cast<const ForwardResponse*>(
               &_ForwardResponse_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    4;

  void Swap(ForwardResponse* other);
  friend void swap(ForwardResponse& a, ForwardResponse& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline ForwardResponse* New() const final {
    return CreateMaybeMessage<ForwardResponse>(NULL);
  }

  ForwardResponse* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<ForwardResponse>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const ForwardResponse& from);
  void MergeFrom(const ForwardResponse& from);
  void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) final;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const final;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(ForwardResponse* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  typedef ForwardResponse_Box Box;
  typedef ForwardResponse_Label Label;

  // accessors -------------------------------------------------------

  // repeated .inference.wa.ForwardResponse.Box boxes = 5;
  int boxes_size() const;
  void clear_boxes();
  static const int kBoxesFieldNumber = 5;
  ::inference::wa::ForwardResponse_Box* mutable_boxes(int index);
  ::google::protobuf::RepeatedPtrField< ::inference::wa::ForwardResponse_Box >*
      mutable_boxes();
  const ::inference::wa::ForwardResponse_Box& boxes(int index) const;
  ::inference::wa::ForwardResponse_Box* add_boxes();
  const ::google::protobuf::RepeatedPtrField< ::inference::wa::ForwardResponse_Box >&
      boxes() const;

  // repeated .inference.wa.ForwardResponse.Label label = 6;
  int label_size() const;
  void clear_label();
  static const int kLabelFieldNumber = 6;
  ::inference::wa::ForwardResponse_Label* mutable_label(int index);
  ::google::protobuf::RepeatedPtrField< ::inference::wa::ForwardResponse_Label >*
      mutable_label();
  const ::inference::wa::ForwardResponse_Label& label(int index) const;
  ::inference::wa::ForwardResponse_Label* add_label();
  const ::google::protobuf::RepeatedPtrField< ::inference::wa::ForwardResponse_Label >&
      label() const;

  // string message = 2;
  void clear_message();
  static const int kMessageFieldNumber = 2;
  const ::std::string& message() const;
  void set_message(const ::std::string& value);
  #if LANG_CXX11
  void set_message(::std::string&& value);
  #endif
  void set_message(const char* value);
  void set_message(const char* value, size_t size);
  ::std::string* mutable_message();
  ::std::string* release_message();
  void set_allocated_message(::std::string* message);

  // int32 code = 1;
  void clear_code();
  static const int kCodeFieldNumber = 1;
  ::google::protobuf::int32 code() const;
  void set_code(::google::protobuf::int32 value);

  // @@protoc_insertion_point(class_scope:inference.wa.ForwardResponse)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::RepeatedPtrField< ::inference::wa::ForwardResponse_Box > boxes_;
  ::google::protobuf::RepeatedPtrField< ::inference::wa::ForwardResponse_Label > label_;
  ::google::protobuf::internal::ArenaStringPtr message_;
  ::google::protobuf::int32 code_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::protobuf_forward_2eproto::TableStruct;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// ForwardRequest_Data

// bytes body = 3;
inline void ForwardRequest_Data::clear_body() {
  body_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline const ::std::string& ForwardRequest_Data::body() const {
  // @@protoc_insertion_point(field_get:inference.wa.ForwardRequest.Data.body)
  return body_.GetNoArena();
}
inline void ForwardRequest_Data::set_body(const ::std::string& value) {
  
  body_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:inference.wa.ForwardRequest.Data.body)
}
#if LANG_CXX11
inline void ForwardRequest_Data::set_body(::std::string&& value) {
  
  body_.SetNoArena(
    &::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::move(value));
  // @@protoc_insertion_point(field_set_rvalue:inference.wa.ForwardRequest.Data.body)
}
#endif
inline void ForwardRequest_Data::set_body(const char* value) {
  GOOGLE_DCHECK(value != NULL);
  
  body_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:inference.wa.ForwardRequest.Data.body)
}
inline void ForwardRequest_Data::set_body(const void* value, size_t size) {
  
  body_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:inference.wa.ForwardRequest.Data.body)
}
inline ::std::string* ForwardRequest_Data::mutable_body() {
  
  // @@protoc_insertion_point(field_mutable:inference.wa.ForwardRequest.Data.body)
  return body_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* ForwardRequest_Data::release_body() {
  // @@protoc_insertion_point(field_release:inference.wa.ForwardRequest.Data.body)
  
  return body_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void ForwardRequest_Data::set_allocated_body(::std::string* body) {
  if (body != NULL) {
    
  } else {
    
  }
  body_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), body);
  // @@protoc_insertion_point(field_set_allocated:inference.wa.ForwardRequest.Data.body)
}

// -------------------------------------------------------------------

// ForwardRequest

// .inference.wa.ForwardRequest.Data data = 1;
inline bool ForwardRequest::has_data() const {
  return this != internal_default_instance() && data_ != NULL;
}
inline void ForwardRequest::clear_data() {
  if (GetArenaNoVirtual() == NULL && data_ != NULL) {
    delete data_;
  }
  data_ = NULL;
}
inline const ::inference::wa::ForwardRequest_Data& ForwardRequest::_internal_data() const {
  return *data_;
}
inline const ::inference::wa::ForwardRequest_Data& ForwardRequest::data() const {
  const ::inference::wa::ForwardRequest_Data* p = data_;
  // @@protoc_insertion_point(field_get:inference.wa.ForwardRequest.data)
  return p != NULL ? *p : *reinterpret_cast<const ::inference::wa::ForwardRequest_Data*>(
      &::inference::wa::_ForwardRequest_Data_default_instance_);
}
inline ::inference::wa::ForwardRequest_Data* ForwardRequest::release_data() {
  // @@protoc_insertion_point(field_release:inference.wa.ForwardRequest.data)
  
  ::inference::wa::ForwardRequest_Data* temp = data_;
  data_ = NULL;
  return temp;
}
inline ::inference::wa::ForwardRequest_Data* ForwardRequest::mutable_data() {
  
  if (data_ == NULL) {
    auto* p = CreateMaybeMessage<::inference::wa::ForwardRequest_Data>(GetArenaNoVirtual());
    data_ = p;
  }
  // @@protoc_insertion_point(field_mutable:inference.wa.ForwardRequest.data)
  return data_;
}
inline void ForwardRequest::set_allocated_data(::inference::wa::ForwardRequest_Data* data) {
  ::google::protobuf::Arena* message_arena = GetArenaNoVirtual();
  if (message_arena == NULL) {
    delete data_;
  }
  if (data) {
    ::google::protobuf::Arena* submessage_arena = NULL;
    if (message_arena != submessage_arena) {
      data = ::google::protobuf::internal::GetOwnedMessage(
          message_arena, data, submessage_arena);
    }
    
  } else {
    
  }
  data_ = data;
  // @@protoc_insertion_point(field_set_allocated:inference.wa.ForwardRequest.data)
}

// int32 h = 5;
inline void ForwardRequest::clear_h() {
  h_ = 0;
}
inline ::google::protobuf::int32 ForwardRequest::h() const {
  // @@protoc_insertion_point(field_get:inference.wa.ForwardRequest.h)
  return h_;
}
inline void ForwardRequest::set_h(::google::protobuf::int32 value) {
  
  h_ = value;
  // @@protoc_insertion_point(field_set:inference.wa.ForwardRequest.h)
}

// int32 w = 6;
inline void ForwardRequest::clear_w() {
  w_ = 0;
}
inline ::google::protobuf::int32 ForwardRequest::w() const {
  // @@protoc_insertion_point(field_get:inference.wa.ForwardRequest.w)
  return w_;
}
inline void ForwardRequest::set_w(::google::protobuf::int32 value) {
  
  w_ = value;
  // @@protoc_insertion_point(field_set:inference.wa.ForwardRequest.w)
}

// -------------------------------------------------------------------

// ForwardResponse_Box

// float xmin = 1;
inline void ForwardResponse_Box::clear_xmin() {
  xmin_ = 0;
}
inline float ForwardResponse_Box::xmin() const {
  // @@protoc_insertion_point(field_get:inference.wa.ForwardResponse.Box.xmin)
  return xmin_;
}
inline void ForwardResponse_Box::set_xmin(float value) {
  
  xmin_ = value;
  // @@protoc_insertion_point(field_set:inference.wa.ForwardResponse.Box.xmin)
}

// float ymin = 2;
inline void ForwardResponse_Box::clear_ymin() {
  ymin_ = 0;
}
inline float ForwardResponse_Box::ymin() const {
  // @@protoc_insertion_point(field_get:inference.wa.ForwardResponse.Box.ymin)
  return ymin_;
}
inline void ForwardResponse_Box::set_ymin(float value) {
  
  ymin_ = value;
  // @@protoc_insertion_point(field_set:inference.wa.ForwardResponse.Box.ymin)
}

// float xmax = 3;
inline void ForwardResponse_Box::clear_xmax() {
  xmax_ = 0;
}
inline float ForwardResponse_Box::xmax() const {
  // @@protoc_insertion_point(field_get:inference.wa.ForwardResponse.Box.xmax)
  return xmax_;
}
inline void ForwardResponse_Box::set_xmax(float value) {
  
  xmax_ = value;
  // @@protoc_insertion_point(field_set:inference.wa.ForwardResponse.Box.xmax)
}

// float ymax = 4;
inline void ForwardResponse_Box::clear_ymax() {
  ymax_ = 0;
}
inline float ForwardResponse_Box::ymax() const {
  // @@protoc_insertion_point(field_get:inference.wa.ForwardResponse.Box.ymax)
  return ymax_;
}
inline void ForwardResponse_Box::set_ymax(float value) {
  
  ymax_ = value;
  // @@protoc_insertion_point(field_set:inference.wa.ForwardResponse.Box.ymax)
}

// float score = 5;
inline void ForwardResponse_Box::clear_score() {
  score_ = 0;
}
inline float ForwardResponse_Box::score() const {
  // @@protoc_insertion_point(field_get:inference.wa.ForwardResponse.Box.score)
  return score_;
}
inline void ForwardResponse_Box::set_score(float value) {
  
  score_ = value;
  // @@protoc_insertion_point(field_set:inference.wa.ForwardResponse.Box.score)
}

// int32 label = 6;
inline void ForwardResponse_Box::clear_label() {
  label_ = 0;
}
inline ::google::protobuf::int32 ForwardResponse_Box::label() const {
  // @@protoc_insertion_point(field_get:inference.wa.ForwardResponse.Box.label)
  return label_;
}
inline void ForwardResponse_Box::set_label(::google::protobuf::int32 value) {
  
  label_ = value;
  // @@protoc_insertion_point(field_set:inference.wa.ForwardResponse.Box.label)
}

// -------------------------------------------------------------------

// ForwardResponse_Label

// int32 index = 1;
inline void ForwardResponse_Label::clear_index() {
  index_ = 0;
}
inline ::google::protobuf::int32 ForwardResponse_Label::index() const {
  // @@protoc_insertion_point(field_get:inference.wa.ForwardResponse.Label.index)
  return index_;
}
inline void ForwardResponse_Label::set_index(::google::protobuf::int32 value) {
  
  index_ = value;
  // @@protoc_insertion_point(field_set:inference.wa.ForwardResponse.Label.index)
}

// float score = 2;
inline void ForwardResponse_Label::clear_score() {
  score_ = 0;
}
inline float ForwardResponse_Label::score() const {
  // @@protoc_insertion_point(field_get:inference.wa.ForwardResponse.Label.score)
  return score_;
}
inline void ForwardResponse_Label::set_score(float value) {
  
  score_ = value;
  // @@protoc_insertion_point(field_set:inference.wa.ForwardResponse.Label.score)
}

// -------------------------------------------------------------------

// ForwardResponse

// int32 code = 1;
inline void ForwardResponse::clear_code() {
  code_ = 0;
}
inline ::google::protobuf::int32 ForwardResponse::code() const {
  // @@protoc_insertion_point(field_get:inference.wa.ForwardResponse.code)
  return code_;
}
inline void ForwardResponse::set_code(::google::protobuf::int32 value) {
  
  code_ = value;
  // @@protoc_insertion_point(field_set:inference.wa.ForwardResponse.code)
}

// string message = 2;
inline void ForwardResponse::clear_message() {
  message_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline const ::std::string& ForwardResponse::message() const {
  // @@protoc_insertion_point(field_get:inference.wa.ForwardResponse.message)
  return message_.GetNoArena();
}
inline void ForwardResponse::set_message(const ::std::string& value) {
  
  message_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:inference.wa.ForwardResponse.message)
}
#if LANG_CXX11
inline void ForwardResponse::set_message(::std::string&& value) {
  
  message_.SetNoArena(
    &::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::move(value));
  // @@protoc_insertion_point(field_set_rvalue:inference.wa.ForwardResponse.message)
}
#endif
inline void ForwardResponse::set_message(const char* value) {
  GOOGLE_DCHECK(value != NULL);
  
  message_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:inference.wa.ForwardResponse.message)
}
inline void ForwardResponse::set_message(const char* value, size_t size) {
  
  message_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:inference.wa.ForwardResponse.message)
}
inline ::std::string* ForwardResponse::mutable_message() {
  
  // @@protoc_insertion_point(field_mutable:inference.wa.ForwardResponse.message)
  return message_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* ForwardResponse::release_message() {
  // @@protoc_insertion_point(field_release:inference.wa.ForwardResponse.message)
  
  return message_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void ForwardResponse::set_allocated_message(::std::string* message) {
  if (message != NULL) {
    
  } else {
    
  }
  message_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), message);
  // @@protoc_insertion_point(field_set_allocated:inference.wa.ForwardResponse.message)
}

// repeated .inference.wa.ForwardResponse.Box boxes = 5;
inline int ForwardResponse::boxes_size() const {
  return boxes_.size();
}
inline void ForwardResponse::clear_boxes() {
  boxes_.Clear();
}
inline ::inference::wa::ForwardResponse_Box* ForwardResponse::mutable_boxes(int index) {
  // @@protoc_insertion_point(field_mutable:inference.wa.ForwardResponse.boxes)
  return boxes_.Mutable(index);
}
inline ::google::protobuf::RepeatedPtrField< ::inference::wa::ForwardResponse_Box >*
ForwardResponse::mutable_boxes() {
  // @@protoc_insertion_point(field_mutable_list:inference.wa.ForwardResponse.boxes)
  return &boxes_;
}
inline const ::inference::wa::ForwardResponse_Box& ForwardResponse::boxes(int index) const {
  // @@protoc_insertion_point(field_get:inference.wa.ForwardResponse.boxes)
  return boxes_.Get(index);
}
inline ::inference::wa::ForwardResponse_Box* ForwardResponse::add_boxes() {
  // @@protoc_insertion_point(field_add:inference.wa.ForwardResponse.boxes)
  return boxes_.Add();
}
inline const ::google::protobuf::RepeatedPtrField< ::inference::wa::ForwardResponse_Box >&
ForwardResponse::boxes() const {
  // @@protoc_insertion_point(field_list:inference.wa.ForwardResponse.boxes)
  return boxes_;
}

// repeated .inference.wa.ForwardResponse.Label label = 6;
inline int ForwardResponse::label_size() const {
  return label_.size();
}
inline void ForwardResponse::clear_label() {
  label_.Clear();
}
inline ::inference::wa::ForwardResponse_Label* ForwardResponse::mutable_label(int index) {
  // @@protoc_insertion_point(field_mutable:inference.wa.ForwardResponse.label)
  return label_.Mutable(index);
}
inline ::google::protobuf::RepeatedPtrField< ::inference::wa::ForwardResponse_Label >*
ForwardResponse::mutable_label() {
  // @@protoc_insertion_point(field_mutable_list:inference.wa.ForwardResponse.label)
  return &label_;
}
inline const ::inference::wa::ForwardResponse_Label& ForwardResponse::label(int index) const {
  // @@protoc_insertion_point(field_get:inference.wa.ForwardResponse.label)
  return label_.Get(index);
}
inline ::inference::wa::ForwardResponse_Label* ForwardResponse::add_label() {
  // @@protoc_insertion_point(field_add:inference.wa.ForwardResponse.label)
  return label_.Add();
}
inline const ::google::protobuf::RepeatedPtrField< ::inference::wa::ForwardResponse_Label >&
ForwardResponse::label() const {
  // @@protoc_insertion_point(field_list:inference.wa.ForwardResponse.label)
  return label_;
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------

// -------------------------------------------------------------------

// -------------------------------------------------------------------

// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace wa
}  // namespace inference

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_INCLUDED_forward_2eproto
