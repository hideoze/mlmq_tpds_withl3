// Host-only, ordered CSR export and independent int64 Dijkstra oracle.
#include "../../SSSP/l3/l3_chain_shortcuts.h"
#include <cstdio>
#include <fstream>
#include <queue>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
struct Graph {
    int fd, n, m; size_t bytes; const char *p; const int *dst,*w;
    std::vector<int> row;
    explicit Graph(const char *name) {
        fd=open(name,O_RDONLY); struct stat s{};
        if(fd<0 || fstat(fd,&s)) throw std::runtime_error("open/stat");
        bytes=s.st_size; if(bytes<32) throw std::runtime_error("header");
        p=(const char*)mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,fd,0);
        if(p==MAP_FAILED) throw std::runtime_error("mmap");
        auto h=(const uint64_t*)p;
        if(h[0]!=1 || h[1]!=4 || h[2]>=INT_MAX || h[3]>=INT_MAX) throw std::runtime_error("format");
        n=h[2];m=h[3];
        if(bytes<32+8ull*n+8ull*m+4ull*(m&1)) throw std::runtime_error("truncated");
        auto ends=(const uint64_t*)(p+32); dst=(const int*)(ends+n); w=dst+m+(m&1);
        row.resize(n+1);
        for(int u=0;u<n;++u) {if(ends[u]>uint64_t(m) || (u && ends[u]<ends[u-1]))throw std::runtime_error("rows");row[u+1]=ends[u];}
        if(row.back()!=m) throw std::runtime_error("last row");
        for(int e=0;e<m;++e) if(dst[e]<0 || dst[e]>=n || w[e]<0) throw std::runtime_error("edge");
    }
    ~Graph(){munmap((void*)p,bytes);close(fd);}
};
std::vector<int> oracle(int n,const int *row,const int *dst,const int *w,int source) {
    const long long INF=1ll<<62;
    std::vector<long long>d(n,INF);d.at(source)=0;
    using Entry=std::pair<long long,int>;
    std::priority_queue<Entry,std::vector<Entry>,std::greater<Entry>>q;q.push({0,source});
    while(!q.empty()) {auto [cost,u]=q.top();q.pop();if(cost!=d[u])continue;
        for(int e=row[u];e<row[u+1];++e)if(cost+w[e]<d[dst[e]]){d[dst[e]]=cost+w[e];q.push({d[dst[e]],dst[e]});}}
    std::vector<int> result(n);long long maxd=0;int reached=0;
    for(int u=0;u<n;++u){if(d[u]!=INF && d[u]>=INT_MAX)throw std::runtime_error("unrepresentable distance");result[u]=d[u]==INF?INT_MAX:int(d[u]);if(d[u]!=INF){++reached;maxd=std::max(maxd,d[u]);}}
    printf("ORACLE source=%d vertices=%d reached=%d max_distance=%lld arithmetic=int64\n",source,n,reached,maxd);
    return result;
}
int main(int argc,char **argv)try {
    if(argc!=6)throw std::runtime_error("INPUT SOURCE CUT OUTPUT_GR OUTPUT_ORACLE");
    Graph g(argv[1]);int source=std::stoi(argv[2]),cut=std::stoi(argv[3]);
    auto d=oracle(g.n,g.row.data(),g.dst,g.w,source);
    l3_chain_shortcut_view v;v.build(g.n,g.m,g.row.data(),g.dst,g.w,cut,source);
    if(d!=oracle(g.n,v.rows.data(),v.destinations.data(),v.weights.data(),source))throw std::runtime_error("G/G+ distance mismatch");
    std::ofstream f(argv[4],std::ios::binary);uint64_t h[]={1,4,uint64_t(g.n),v.destinations.size()};
    f.write((char*)h,sizeof(h));for(int u=1;u<=g.n;++u){uint64_t end=v.rows[u];f.write((char*)&end,8);}
    f.write((char*)v.destinations.data(),4*v.destinations.size());int zero=0;if(v.destinations.size()&1)f.write((char*)&zero,4);
    f.write((char*)v.weights.data(),4*v.weights.size());f.close();if(!f)throw std::runtime_error("export write");
    Graph back(argv[4]);
    if(back.row!=v.rows || !std::equal(v.destinations.begin(),v.destinations.end(),back.dst) || !std::equal(v.weights.begin(),v.weights.end(),back.w))throw std::runtime_error("CSR readback");
    std::ofstream ref(argv[5],std::ios::binary);ref.write((char*)d.data(),4*d.size());ref.close();if(!ref)throw std::runtime_error("oracle write");
    printf("EXPORT vertices=%d original_edges=%d augmented_edges=%d cut=%d source=%d shortcuts=%d ordered_csr_readback=PASS\n",g.n,g.m,back.m,cut,source,v.shortcuts);
    return 0;
}catch(const std::exception&e){fprintf(stderr,"EXPORT_FAIL %s\n",e.what());return 2;}
