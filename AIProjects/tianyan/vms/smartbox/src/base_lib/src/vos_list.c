#define __LIST_C__

#include "vos_list.h"

void vos_list_init(vos_list_type_t * node)
{
    ((vos_list_t*)node)->next = ((vos_list_t*)node)->prev = node;
}

int vos_list_empty(const vos_list_type_t * node)
{
    return ((vos_list_t*)node)->next == node;
}


void vos_list_push_back(vos_list_type_t *list, vos_list_type_t *node)
{
    vos_list_insert_before(list, node);
}

void vos_list_push_front(vos_list_type_t *list, vos_list_type_t *node)
{
    vos_list_insert_after(list, node);
}

vos_list_type_t* vos_list_pop_front(vos_list_type_t *list)
{
	vos_list_type_t* node;
	node = ((vos_list_t*)list)->next;
	vos_list_erase(node);
	return node;
}

void vos_link_node(vos_list_type_t *prev, vos_list_type_t *next)
{
    ((vos_list_t*)prev)->next = next;
    ((vos_list_t*)next)->prev = prev;
}

void vos_list_insert_after(vos_list_type_t *pos, vos_list_type_t *node)
{
    ((vos_list_t*)node)->prev = pos;
    ((vos_list_t*)node)->next = ((vos_list_t*)pos)->next;
    ((vos_list_t*) ((vos_list_t*)pos)->next)->prev = node;
    ((vos_list_t*)pos)->next = node;
}


void vos_list_insert_before(vos_list_type_t *pos, vos_list_type_t *node)
{
    vos_list_insert_after(((vos_list_t*)pos)->prev, node);
}


void vos_list_insert_nodes_after(vos_list_type_t *pos, vos_list_type_t *lst)
{
    vos_list_t *lst_last = (vos_list_t *) ((vos_list_t*)lst)->prev;
    vos_list_t *pos_next = (vos_list_t *) ((vos_list_t*)pos)->next;

    vos_link_node(pos, lst);
    vos_link_node(lst_last, pos_next);
}

void vos_list_insert_nodes_before(vos_list_type_t *pos, vos_list_type_t *lst)
{
    vos_list_insert_nodes_after(((vos_list_t*)pos)->prev, lst);
}

void vos_list_merge_last(vos_list_type_t *list1, vos_list_type_t *list2)
{
    if (!vos_list_empty(list2))
    {
    	vos_link_node(((vos_list_t*)list1)->prev, ((vos_list_t*)list2)->next);
    	vos_link_node(((vos_list_t*)list2)->prev, list1);
    	vos_list_init(list2);
    }
}

void vos_list_merge_first(vos_list_type_t *list1, vos_list_type_t *list2)
{
    if (!vos_list_empty(list2))
    {
    	vos_link_node(((vos_list_t*)list2)->prev, ((vos_list_t*)list1)->next);
    	vos_link_node(((vos_list_t*)list1), ((vos_list_t*)list2)->next);
    	vos_list_init(list2);
    }
}

void vos_list_erase(vos_list_type_t *node)
{
    vos_link_node( ((vos_list_t*)node)->prev, ((vos_list_t*)node)->next);
    vos_list_init(node);
}


vos_list_type_t* vos_list_find_node(vos_list_type_t *list, vos_list_type_t *node)
{
    vos_list_t *p = (vos_list_t *) ((vos_list_t*)list)->next;
    
    while (p != list && p != node)
	    p = (vos_list_t *) p->next;

    return p==node ? p : NULL;
}


vos_list_type_t* vos_list_search(vos_list_type_t *list, void *value,
	       		int (*comp)(void *value, const vos_list_type_t *node))
{
    vos_list_t *p = (vos_list_t *) ((vos_list_t*)list)->next;
    
    while (p != list && (*comp)(value, p) != 0)
	    p = (vos_list_t *) p->next;

    return p==list ? NULL : p;
}


vos_size_t vos_list_size(const vos_list_type_t *list)
{
    const vos_list_t *node = (const vos_list_t*) ((const vos_list_t*)list)->next;
    vos_size_t count = 0;

    while (node != list) 
    {
    	++count;
    	node = (vos_list_t*)node->next;
    }

    return count;
}



