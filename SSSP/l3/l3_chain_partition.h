#pragma once
// Topology-only micro-partitions; never constructs an augmented CSR or reads
// query distances. Each route expands one chain closure at GPU runtime.
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <vector>
struct l3_chain_partition_node { int id, forward, reverse, length; };
struct l3_chain_partition_index {
    std::vector<uint64_t> route;
    std::vector<l3_chain_partition_node> nodes;
    int segments=0, interiors=0, overflow=0;
    void build(int n,const int*row,const int*dst,const int*w,int begin,int end) {
        if(n<=0||begin<0||end>n||begin>=end||row[0]!=0)throw std::runtime_error("chain partition dimensions");
        route.assign(end-begin+1,0);nodes.clear();segments=interiors=overflow=0;
        std::vector<unsigned char> eligible(end-begin),seen(end-begin);
        std::vector<int> incoming(end-begin),reverse_first(end-begin),reverse_second(end-begin);
        for(int u=begin;u<end;++u)if(row[u+1]-row[u]==2){
            int a=dst[row[u]],b=dst[row[u]+1];
            eligible[u-begin]=a!=u&&b!=u&&a!=b&&a>=begin&&a<end&&b>=begin&&b<end;
        }
        for(int u=0;u<n;++u)for(int e=row[u];e<row[u+1];++e){
            int v=dst[e];if(v<0||v>=n||w[e]<0)throw std::runtime_error("chain partition edge");
            if(v>=begin&&v<end&&eligible[v-begin]){
                const int local=v-begin;
                ++incoming[local];
                if(dst[row[v]]==u)++reverse_first[local];
                if(dst[row[v]+1]==u)++reverse_second[local];
            }
        }
        // A strict interior has exactly one reciprocal edge from each of its
        // two distinct neighbours.  Merely counting two matching sources can
        // accept parallel edges from one neighbour and later abort the whole
        // index when the missing reverse edge is encountered.  Such vertices
        // must stay on the ordinary CSR path instead.
        for(int u=begin;u<end;++u){
            const int local=u-begin;
            eligible[local]=eligible[local]&&incoming[local]==2&&
                            reverse_first[local]==1&&reverse_second[local]==1;
        }
        for(int u=begin;u<end;++u)if(!eligible[u-begin])for(int first=row[u];first<row[u+1];++first){
            int v=dst[first];if(v<begin||v>=end||!eligible[v-begin]||seen[v-begin])continue;
            std::vector<l3_chain_partition_node> path{{u+1,0,0,0}};
            long long f=0,r=0;int prior=u,edge=first;bool bad=false;
            while(true){
                f+=w[edge];int back=-1;
                for(int e=row[v];e<row[v+1];++e)if(dst[e]==prior){back=e;break;}
                if(back<0)throw std::runtime_error("chain missing reciprocal edge");
                r+=w[back];bad|=f>=INT_MAX||r>=INT_MAX;
                path.push_back({v+1,bad?0:int(f),bad?0:int(r),0});
                if(!eligible[v-begin])break;
                if(seen[v-begin])throw std::runtime_error("chain revisited interior");
                seen[v-begin]=1;
                int next=row[v]+(dst[row[v]]==prior);prior=v;v=dst[next];edge=next;
            }
            if(bad){++overflow;continue;}
            if(path.size()<4||v==u)continue; // singleton/cycle stays on original CSR
            if(nodes.size()+path.size()>=INT_MAX)throw std::runtime_error("chain partition index overflow");
            int base=nodes.size();path[0].length=path.size();
            nodes.insert(nodes.end(),path.begin(),path.end());++segments;interiors+=path.size()-2;
            for(int i=1;i<int(path.size())-1;++i)
                route[path[i].id-begin]=(uint64_t(base+1)<<32)|uint32_t(base+i);
        }
    }
};
