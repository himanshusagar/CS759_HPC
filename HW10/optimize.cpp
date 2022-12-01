#include "optimize.h"

size_t vec_length(const vec *v)
{
    return v->len;
}
data_t *get_vec_start(vec *v)
{
    if(v == nullptr)
        return nullptr;
    return v->data;
}
void optimize1(vec *v, data_t *dest)
{
    int length = vec_length(v);
    data_t *d = get_vec_start(v);
    data_t temp = IDENT;
    for(int i = 0; i < length; i++)
        temp = temp OP d[i];
    *dest = temp;
}

void optimize2(vec *v, data_t *dest)
{
    int length = vec_length(v);
    int limit = length - 1;
    data_t *d = get_vec_start(v);
    data_t x = IDENT;
    int i;
    for(i = 0; i < limit; i += 2) 
    {
        x = (x OP d[i]) OP d[i + 1];
    }
    for(; i < length; i++) 
    {
        x = x OP d[i];
    }
    *dest = x;
}

void optimize3(vec *v, data_t *dest)
{
    int length = vec_length(v);
    int limit = length - 1;
    data_t *d = get_vec_start(v);
    data_t x = IDENT;
    int i;
    for(i = 0; i < limit; i += 2) 
    {
        x = x OP(d[i] OP d[i + 1]);
    }
    for(; i < length; i++) 
    {
        x = x OP d[i];
    }
    *dest = x;
}

void optimize4(vec *v, data_t *dest)
{
    int length = vec_length(v);
    int limit = length - 1;
    data_t *d = get_vec_start(v);
    data_t x0 = IDENT;
    data_t x1 = IDENT;
    int i;
    for(i = 0; i < limit; i += 2) 
    {
        x0 = x0 OP d[i];
        x1 = x1 OP d[i + 1];
    }
    for(; i < length; i++) 
    {
        x0 = x0 OP d[i];
    }
    *dest = x0 OP x1;
}

void optimize5(vec *v, data_t *dest)
{
    int length = vec_length(v);
    int limit = length - 2;
    data_t *d = get_vec_start(v);
    data_t temp0 = IDENT;
    data_t temp1 = IDENT;
    data_t temp2 = IDENT;
    int i;
    for(i = 0; i < limit; i += 3)
    {
        temp0 = temp0 OP d[i];
        temp1 = temp1 OP d[i + 1];
        temp2 = temp2 OP d[i + 2];
    }
    for(; i < length; i++) 
    {
        temp0 = temp0 OP d[i];
    }
    *dest = temp0 OP temp1 OP temp2;
}