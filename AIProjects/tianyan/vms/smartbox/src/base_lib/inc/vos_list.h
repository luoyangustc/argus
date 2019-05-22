#ifndef __VOS_LIST_H__
#define __VOS_LIST_H__

#undef  EXT
#ifndef __LIST_C__
#define EXT extern
#else
#define EXT
#endif

#include "vos_types.h"

#define VOS_DECL_LIST_MEMBER(type) \
                        type *prev; \
                        type *next

struct vos_list_t
{
    VOS_DECL_LIST_MEMBER(void);
};

EXT void vos_list_init(vos_list_type_t * node);
EXT int vos_list_empty(const vos_list_type_t * node);
EXT void vos_list_insert_before(vos_list_type_t *pos, vos_list_type_t *node);
EXT void vos_list_push_back(vos_list_type_t *list, vos_list_type_t *node);
EXT void vos_list_insert_nodes_before(vos_list_type_t *lst, vos_list_type_t *nodes);
EXT void vos_list_insert_after(vos_list_type_t *pos, vos_list_type_t *node);
EXT void vos_list_push_front(vos_list_type_t *list, vos_list_type_t *node);
EXT vos_list_type_t* vos_list_pop_front(vos_list_type_t *list);
EXT void vos_list_insert_nodes_after(vos_list_type_t *lst, vos_list_type_t *nodes);
EXT void vos_list_merge_first(vos_list_type_t *list1, vos_list_type_t *list2);
EXT void vos_list_merge_last( vos_list_type_t *list1, vos_list_type_t *list2);
EXT void vos_list_erase(vos_list_type_t *node);
EXT vos_list_type_t* vos_list_find_node(vos_list_type_t *list,  vos_list_type_t *node);
EXT vos_list_type_t* vos_list_search(vos_list_type_t *list, void *value,
				       int (*comp)(void *value, 
					   const vos_list_type_t *node)
				       );
EXT vos_size_t vos_list_size(const vos_list_type_t *list);


#endif	/* __VOS_LIST_H__ */


