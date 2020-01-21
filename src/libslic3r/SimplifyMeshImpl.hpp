#ifndef SIMPLIFYMESHIMPL_HPP
#define SIMPLIFYMESHIMPL_HPP

#include <random>
#include <vector>
#include <array>
#include <type_traits>
#include <algorithm>

namespace SimplifyMesh {

using Bary = std::array<double, 3>;

template<class Vertex> struct vertex_traits {
    using coord_type = typename Vertex::coord_type;
    using compute_type = coord_type;
    
    static coord_type x(const Vertex &v);
    static coord_type& x(Vertex &v);
    
    static coord_type y(const Vertex &v);
    static coord_type& y(Vertex &v);
    
    static coord_type z(const Vertex &v);
    static coord_type& z(Vertex &v);
    
    static Vertex interpolate(const Bary &                 b,
                              const std::array<Vertex, 3> &triang)
    {
        return Vertex{triang[0] * b[0], triang[1] * b[1], triang[2] * b[2]};
    }
};

template<class Mesh> struct mesh_traits {
    using vertex_t = typename Mesh::vertex_t;
    static size_t face_count(const Mesh &m);
    static size_t vertex_count(const Mesh &m);
    static vertex_t vertex(const Mesh &m, size_t vertex_idx);
    static std::array<size_t, 3> triangle(const Mesh &m, size_t face_idx);
};

namespace implementation {

// A shorter C++14 style form of the enable_if metafunction
template<bool B, class T>
using enable_if_t = typename std::enable_if<B, T>::type;

// Meta predicates for floating, 'scaled coord' and generic arithmetic types
template<class T, class O = T>
using FloatingOnly = enable_if_t<std::is_floating_point<T>::value, O>;

template<class T, class O = T>
using IntegerOnly = enable_if_t<std::is_integral<T>::value, O>;

template<class T, class O = T>
using ArithmeticOnly = enable_if_t<std::is_arithmetic<T>::value, O>;

template<class T> FloatingOnly<T, bool> is_approx(T val, T ref) { return std::abs(val - ref) < 1e-8; }
template<class T> IntegerOnly <T, bool> is_approx(T val, T ref) { val == ref; }

template<class T, size_t N = 10> class SymetricMatrix {
public:
    
    explicit SymetricMatrix(ArithmeticOnly<T> c = T()) { std::fill(m, m + N, c); }
    
    SymetricMatrix(T m11, T m12, T m13, T m14,
                   T m22, T m23, T m24,
                   T m33, T m34,
                   T m44)
    {
        m[0] = m11;  m[1] = m12;  m[2] = m13;  m[3] = m14;
        m[4] = m22;  m[5] = m23;  m[6] = m24;
        m[7] = m33;  m[8] = m34;
        m[9] = m44;
    }
    
    // Make plane
    SymetricMatrix(T a, T b, T c, T d)
    {
        m[0] = a * a; m[1] = a * b; m[2] = a * c; m[3] = a * d;
        m[4] = b * b; m[5] = b * c; m[6] = b * d;
        m[7] = c * c; m[8] = c * d;
        m[9] = d * d;
    }
    
    T operator[](int c) const { return m[c]; }
    
    // Determinant
    T det(int a11, int a12, int a13,
          int a21, int a22, int a23,
          int a31, int a32, int a33)
    {
        T det = m[a11] * m[a22] * m[a33] + m[a13] * m[a21] * m[a32] +
                m[a12] * m[a23] * m[a31] - m[a13] * m[a22] * m[a31] -
                m[a11] * m[a23] * m[a32] - m[a12] * m[a21] * m[a33];
        
        return det;
    }
    
    const SymetricMatrix operator+(const SymetricMatrix& n) const
    {
        return SymetricMatrix(m[0] + n[0], m[1] + n[1], m[2] + n[2], m[3]+n[3],
                              m[4] + n[4], m[5] + n[5], m[6] + n[6], 
                              m[7] + n[7], m[8] + n[8],
                              m[9] + n[9]);
    }
    
    SymetricMatrix& operator+=(const SymetricMatrix& n)
    {
        m[0]+=n[0]; m[1]+=n[1]; m[2]+=n[2]; m[3]+=n[3];
        m[4]+=n[4]; m[5]+=n[5]; m[6]+=n[6]; m[7]+=n[7];
        m[8]+=n[8]; m[9]+=n[9];
        
        return *this;
    }
    
    T m[N];
};

template<class V> using TCoord = typename vertex_traits<V>::coord_type;
template<class V> using TCompute = typename vertex_traits<V>::compute_type;
template<class V> inline TCoord<V> x(const V &v) { return vertex_traits<V>::x(v); }
template<class V> inline TCoord<V> y(const V &v) { return vertex_traits<V>::y(v); }
template<class V> inline TCoord<V> z(const V &v) { return vertex_traits<V>::z(v); }
template<class M> using TVertex = typename mesh_traits<M>::vertex_t;
template<class Mesh> using TMeshCoord = TCoord<TVertex<Mesh>>;

template<class Vertex> TCompute<Vertex> dot(const Vertex &v1, const Vertex &v2)
{
    return TCompute<Vertex>(x(v1)) * x(v2) +
           TCompute<Vertex>(y(v1)) * y(v2) +
           TCompute<Vertex>(z(v1)) * z(v2);
}

template<class Vertex> Vertex cross(const Vertex &a, const Vertex &b)
{
    return Vertex{y(a) * z(b) - z(a) * y(b),
                  z(a) * x(b) - x(a) * z(b),
                  x(a) * y(b) - y(a) * x(b)};
}

template<class Vertex> TCompute<Vertex> lengthsq(const Vertex &v)
{
    return TCompute<Vertex>(x(v)) * x(v) + TCompute<Vertex>(y(v)) * y(v) +
           TCompute<Vertex>(z(v)) * z(v);
}

template<class Vertex> void normalize(Vertex &v)
{
    double square = std::sqrt(lengthsq(v));
    x(v) /= square; y(v) /= square; z(v) /= square;
}

using Bary = std::array<double, 3>;

template<class Vertex>
Bary barycentric(const Vertex &p, const Vertex &a, const Vertex &b, const Vertex &c)
{
    Vertex v0 = (b - a);
    Vertex v1 = (c - a);
    Vertex v2 = (p - a);

    double d00   = dot(v0, v0);
    double d01   = dot(v0, v1);
    double d11   = dot(v1, v1);
    double d20   = dot(v2, v0);
    double d21   = dot(v2, v1);
    double denom = d00 * d11 - d01 * d01;
    double v     = (d11 * d20 - d01 * d21) / denom;
    double w     = (d00 * d21 - d01 * d20) / denom;
    double u     = 1.0 - v - w;

    return {u, v, w};
}

template<class Mesh> class SimplifiableMesh {
    const Mesh *m_mesh;

    using Vertex     = TVertex<Mesh>;
    using Coord      = TMeshCoord<Mesh>;
    using HiPrecison = TCompute<TVertex<Mesh>>;
    using SymMat     = SymetricMatrix<HiPrecison>;

    struct FaceInfo {
        size_t idx;
        double err[4] = {0.};
        bool   deleted = false, dirty = false;
        Vertex n;
        explicit FaceInfo(size_t fi): idx(fi) {}
    };

    struct VertexInfo {
        size_t idx;
        int    tstart, tcount, border;
        SymMat q;
        explicit VertexInfo(size_t vi): idx(vi) {}
    };
    
    struct Ref { size_t face; size_t vertex; };
    
    std::vector<Ref> m_refs;
    std::vector<FaceInfo> m_faceinfo;
    std::vector<VertexInfo> m_vertexinfo;
    
    std::mt19937 m_rng;
    
    void compact_faces();
    void compact();
    
    size_t mesh_vcount() const { return mesh_traits<Mesh>::vertex_count(*m_mesh); }
    size_t mesh_facecount() const { return mesh_traits<Mesh>::face_count(*m_mesh); }
    
    size_t vcount() const { return m_vertexinfo.size(); }
    size_t facecount() const { return m_faceinfo.size(); }
    
    inline Vertex vertex(size_t vi) const
    {
        return mesh_traits<Mesh>::vertex(m_mesh, vi);
    }
    
    inline Vertex vertex(const VertexInfo &vinf) const
    {
        return vertex(vinf.idx);
    }
    
    inline std::array<size_t, 3> triangle(size_t fi) const
    {
        return mesh_traits<Mesh>::triangle(m_mesh, fi);
    }
    
    inline std::array<size_t, 3> triangle(const FaceInfo &finf) const
    {
        return triangle(finf.idx);
    }
    
    inline std::array<Vertex, 3>
    triangle_vertices(const std::array<size_t, 3> &f) const
    {
        std::array<Vertex, 3> p;
        for (size_t i = 0; i < 3; ++i) p[i] = vertex(f[i]);
        return p;
    }

    // Error between vertex and Quadric
    static double vertex_error(const SymMat &q, Coord x, Coord y, Coord z)
    {
        return q[0]* x * x + 2 * q[1] * x * y + 2 * q[2] * x * z +
               2 * q[3] * x + q[4] * y * y + 2 * q[5] * y * z + 2 * q[6] * y +
               q[7] * z * z + 2 * q[8] * z + q[9];
    }
    
    // Error for one edge    
    double calculate_error(size_t id_v1, size_t id_v2, Vertex &p_result);
    
    void calculate_error(FaceInfo &t)
    {
        Vertex p;
        for (size_t j = 0; j < 3; ++j)
            t.err[j] = calculate_error(t.v[j], t.v[(j + 1) % 3], p);
        
        t.err[3] = std::min(t.err[0], std::min(t.err[1],t.err[2]));
    }
    
    void update_mesh(int iteration);
    
public:
    
    explicit SimplifiableMesh(const Mesh *m) : m_mesh{m}
    {
        static_assert(
            std::is_arithmetic<Coord>::value,
            "Coordinate type of mesh has to be an arithmetic type!");
        
        m_faceinfo.reserve(mesh_traits<Mesh>::face_count(*m));
        m_vertexinfo.reserve(mesh_traits<Mesh>::vertex_count(*m));
        for (size_t i = 0; mesh_facecount(); ++i) m_faceinfo.emplace_back(i);
        for (size_t i = 0; mesh_vcount(); ++i) m_vertexinfo.emplace_back(i);
        
        std::random_device rd;
        m_rng.seed(rd());
    }
    
    void seed(long s) { m_rng.seed(std::mt19937::result_type(s)); }
};


template<class Mesh> void SimplifiableMesh<Mesh>::compact_faces()
{
    auto it = std::remove_if(m_faceinfo.begin(), m_faceinfo.end(),
                             [](const FaceInfo &inf) { return inf.deleted; });
    
    m_faceinfo.erase(it, m_faceinfo.end());
}

template<class M> void SimplifiableMesh<M>::compact()
{   
    for (auto &vi : m_vertexinfo) vi.tcount = 0;
    
//    int dst=0;
//    for (size_t i = 0; i < facecount(); ++i)
//        if(!triangles[i].deleted)
//    {
//        Triangle &t=triangles[i];
//        triangles[dst++]=t;
//        loopj(0,3)vertices[t.v[j]].tcount=1;
//    }
    
//    triangles.resize(dst);
    
//    triangles.resize(dst);
//    dst=0;
//    loopi(0,vertices.size())
//        if(vertices[i].tcount)
//    {
//        vertices[i].tstart=dst;
//        vertices[dst].p=vertices[i].p;
//        dst++;
//    }
//    loopi(0,triangles.size())
//    {
//        Triangle &t=triangles[i];
//        loopj(0,3)t.v[j]=vertices[t.v[j]].tstart;
//    }
//    vertices.resize(dst);
}

template<class Mesh>
double SimplifiableMesh<Mesh>::calculate_error(size_t id_v1, size_t id_v2, Vertex &p_result)
{
    // compute interpolated vertex
    
    SymMat q = m_vertexinfo[id_v1].q + m_vertexinfo[id_v2].q;
    
    bool border = m_vertexinfo[id_v1].border & m_vertexinfo[id_v2].border;
    double     error = 0;
    HiPrecison det   = q.det(0, 1, 2, 1, 4, 5, 2, 5, 7);
    
    if (!is_approx(det, HiPrecison(0)) && !border)
    {
        // q_delta is invertible
        p_result.x = Coord(-1) / det * q.det(1, 2, 3, 4, 5, 6, 5, 7, 8);	// vx = A41/det(q_delta)
        p_result.y = Coord( 1) / det * q.det(0, 2, 3, 1, 5, 6, 2, 7, 8);	// vy = A42/det(q_delta)
        p_result.z = Coord(-1) / det * q.det(0, 1, 3, 1, 4, 6, 2, 5, 8);	// vz = A43/det(q_delta)
        
        error = vertex_error(q, p_result.x, p_result.y, p_result.z);
    } else {
        // det = 0 -> try to find best result
        Vertex p1     = vertex(id_v1);
        Vertex p2     = vertex(id_v2);
        Vertex p3     = (p1 + p2) / 2;
        double error1 = vertex_error(q, p1.x, p1.y, p1.z);
        double error2 = vertex_error(q, p2.x, p2.y, p2.z);
        double error3 = vertex_error(q, p3.x, p3.y, p3.z);
        error         = std::min(error1, std::min(error2, error3));

        if (is_approx(error1, error)) p_result = p1;
        if (is_approx(error2, error)) p_result = p2;
        if (is_approx(error3, error)) p_result = p3;
    }

    return error;
}

template<class Mesh> void SimplifiableMesh<Mesh>::update_mesh(int iteration)
{
    if (iteration > 0) compact_faces();
    
    assert(mesh_vcount() == vcount());
        
    //
    // Init Quadrics by Plane & Edge Errors
    //
    // required at the beginning ( iteration == 0 )
    // recomputing during the simplification is not required,
    // but mostly improves the result for closed meshes
    //
    if (iteration == 0) {
                
        for (VertexInfo &vinf : m_vertexinfo) vinf.q = SymMat{};
        for (FaceInfo   &finf : m_faceinfo) {
            std::array<size_t, 3> t = triangle(finf);
            std::array<Vertex, 3> p = triangle_vertices(t);
            Vertex                n = cross(p[1] - p[0], p[2] - p[0]);
            normalize(n);
            finf.n = n;
            
            for (size_t fi : t)
                m_vertexinfo[fi].q += SymMat(x(n), y(n), z(n), -dot(p[0]));
            
            calculate_error(finf);
        }
    }
    
    //
    // Init Quadrics by Plane & Edge Errors
    //
    // required at the beginning ( iteration == 0 )
    // recomputing during the simplification is not required,
    // but mostly improves the result for closed meshes
    //
//    if( iteration == 0 )
//    {
//        loopi(0,vertices.size())
//                vertices[i].q=SymetricMatrix(0.0);
        
//        loopi(0,triangles.size())
//        {
//            Triangle &t=triangles[i];
//            vec3f n,p[3];
//            loopj(0,3) p[j]=vertices[t.v[j]].p;
//            n.cross(p[1]-p[0],p[2]-p[0]);
//            n.normalize();
//            t.n=n;
//            loopj(0,3) vertices[t.v[j]].q =
//                    vertices[t.v[j]].q+SymetricMatrix(n.x,n.y,n.z,-n.dot(p[0]));
//        }
//        loopi(0,triangles.size())
//        {
//            // Calc Edge Error
//            Triangle &t=triangles[i];vec3f p;
//            loopj(0,3) t.err[j]=calculate_error(t.v[j],t.v[(j+1)%3],p);
//            t.err[3]=min(t.err[0],min(t.err[1],t.err[2]));
//        }
//    }
    
    // Init Reference ID list
    for (VertexInfo &vi : m_vertexinfo) { vi.tstart = 0; vi.tcount = 0; }
    
//    loopi(0,vertices.size())
//    {
//        vertices[i].tstart=0;
//        vertices[i].tcount=0;
//    }
    
    
    for (FaceInfo &fi : m_faceinfo)
        for (size_t vidx : triangle(fi)) m_vertexinfo[vidx].tcount++;
    
//    loopi(0,triangles.size())
//    {
//        Triangle &t=triangles[i];
//        loopj(0,3) vertices[t.v[j]].tcount++;
//    }
    
    int tstart = 0;
    for (VertexInfo &vi : m_vertexinfo) {
        vi.tstart = tstart;
        tstart += vi.tcount;
        vi.tcount = 0;
    }

//    int tstart=0;
//    loopi(0,vertices.size())
//    {
//        Vertex &v=vertices[i];
//        v.tstart=tstart;
//        tstart+=v.tcount;
//        v.tcount=0;
//    }
    
    // Write References
    m_refs.resize(m_faceinfo.size() * 3);
    for (size_t i = 0; i < m_faceinfo.size(); ++i) {
        const FaceInfo &fi = m_faceinfo[i];
        std::array<size_t, 3> t = triangle(fi);
        for (size_t j = 0; j < 3; ++j) {
            VertexInfo &vi = m_vertexinfo[t[j]];
            Ref &ref = m_refs[vi.tstart + vi.tcount];
            ref.face = i;
            ref.vertex = j;
            vi.tcount++;
        }
    }
    
//    // Write References
//    refs.resize(triangles.size()*3);
//    loopi(0,triangles.size())
//    {
//        Triangle &t=triangles[i];
//        loopj(0,3)
//        {
//            Vertex &v=vertices[t.v[j]];
//            refs[v.tstart+v.tcount].tid=i;
//            refs[v.tstart+v.tcount].tvertex=j;
//            v.tcount++;
//        }
//    }
    
//    // Identify boundary : vertices[].border=0,1
//    if( iteration == 0 )
//    {
//        std::vector<int> vcount,vids;
        
//        loopi(0,vertices.size())
//                vertices[i].border=0;
        
//        loopi(0,vertices.size())
//        {
//            Vertex &v=vertices[i];
//            vcount.clear();
//            vids.clear();
//            loopj(0,v.tcount)
//            {
//                int k=refs[v.tstart+j].tid;
//                Triangle &t=triangles[k];
//                loopk(0,3)
//                {
//                    int ofs=0,id=t.v[k];
//                    while(ofs<vcount.size())
//                    {
//                        if(vids[ofs]==id)break;
//                        ofs++;
//                    }
//                    if(ofs==vcount.size())
//                    {
//                        vcount.push_back(1);
//                        vids.push_back(id);
//                    }
//                    else
//                        vcount[ofs]++;
//                }
//            }
//            loopj(0,vcount.size()) if(vcount[j]==1)
//                    vertices[vids[j]].border=1;
//        }
//    }
}

} // namespace implementation
} // namespace SimplifyMesh

#endif // SIMPLIFYMESHIMPL_HPP
