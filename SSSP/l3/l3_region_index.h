#pragma once
// Deterministic owner-local BFS regions. Original graph IDs/CSR are unchanged.
// This index stores topology/weights only; all query relaxation stays on GPU.
#include <climits>
#include <stdexcept>
#include <vector>
struct l3_region_index {
    static constexpr int width=32;
    std::vector<int> map,members,rows,sources,weights;
    int regions=0,vertices=0,internal_edges=0;
    void build(int n,const int*row,const int*dst,const int*w,int begin,int end) {
        if(n<=0||begin<0||end>n||begin>=end||row[0]!=0)throw std::runtime_error("region dimensions");
        vertices=end-begin;map.assign(vertices+1,-1);members.clear();regions=0;
        for(int u=0;u<n;++u) {
            if(row[u]>row[u+1])throw std::runtime_error("region CSR order");
            for(int e=row[u];e<row[u+1];++e)
                if(dst[e]<0||dst[e]>=n||w[e]<0)throw std::runtime_error("region edge");
        }
        for(int seed=begin;seed<end;++seed)if(map[seed-begin+1]<0) {
            if(members.size()>size_t(INT_MAX-width))throw std::runtime_error("region padded index overflow");
            const int base=members.size();members.resize(base+width,0);int size=1;
            members[base]=seed+1;map[seed-begin+1]=base;
            for(int k=0;k<size&&size<width;++k) {
                const int u=members[base+k]-1;
                for(int e=row[u];e<row[u+1]&&size<width;++e) {
                    const int v=dst[e];
                    if(v>=begin&&v<end&&map[v-begin+1]<0) {
                        map[v-begin+1]=base+size;members[base+size++]=v+1;
                    }
                }
            }
            ++regions;
        }
        rows.assign(members.size()+1,0);
        for(int u=begin;u<end;++u)for(int e=row[u];e<row[u+1];++e) {
            const int v=dst[e];if(v<begin||v>=end)continue;
            const int a=map[u-begin+1],b=map[v-begin+1];
            if(a/width==b/width)++rows[b+1];
        }
        for(size_t i=1;i<rows.size();++i)rows[i]+=rows[i-1];
        internal_edges=rows.back();sources.resize(internal_edges);weights.resize(internal_edges);
        auto next=rows;
        for(int u=begin;u<end;++u)for(int e=row[u];e<row[u+1];++e) {
            const int v=dst[e];if(v<begin||v>=end)continue;
            const int a=map[u-begin+1],b=map[v-begin+1];
            if(a/width==b/width){int pos=next[b]++;sources[pos]=a%width;weights[pos]=w[e];}
        }
    }
};
