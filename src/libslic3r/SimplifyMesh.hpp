#ifndef MESHSIMPLIFY_HPP
#define MESHSIMPLIFY_HPP

#include <vector>

#include <libslic3r/TriangleMesh.hpp>

namespace Slic3r {

void simplify_mesh(TriangleMesh &);
void simplify_mesh(TriangleMesh &, int face_count, float agressiveness = 0.5f);

}

#endif // MESHSIMPLIFY_H
