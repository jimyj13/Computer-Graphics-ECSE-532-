# Yong-Jung Song 260735417

import helperclasses as hc
import glm
import igl

class Geometry:
    def __init__(self, name: str, gtype: str, materials: list[hc.Material]):
        self.name = name
        self.gtype = gtype
        self.materials = materials

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        return intersect

class Sphere(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], center: glm.vec3, radius: float):
        super().__init__(name, gtype, materials)
        self.center = center
        self.radius = radius
  
    def intersect(self, ray: hc.Ray):
        # TODO: Create intersect code for Sphere

        # Ray-sphere intersection using the quadratic equation
        oc = ray.origin - self.center  # Vector from ray origin to sphere center
        a = glm.dot(ray.direction, ray.direction)
        b = 2.0 * glm.dot(oc, ray.direction)
        c = glm.dot(oc, oc) - self.radius ** 2
        discriminant = b * b - 4.0 * a * c

        #  If the discriminant is negative, there is no intersection
        if discriminant < 0:
            return None

        # Calculate the two possible intersection points t1, t2
        sqrt_discriminant = glm.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2.0 * a)
        t2 = (-b + sqrt_discriminant) / (2.0 * a)

        # Check for the closest valid intersection point
        t = None
        # Take the closest if both t1 and t2 are positive
        if t1 > 0 and t2 > 0:
            t = min(t1, t2)
        # If only t1 is positive, t = t1    
        elif t1 > 0:
            t = t1 
        # If only t2 is positive, t = t2  
        elif t2 > 0:
            t = t2
        # If both are negative, there is no intersection as the intersection is behind the ray
        else:
            return None

        # Calculate intersection point and normal
        intersection_point = ray.getPoint(t)
        normal = glm.normalize(intersection_point - self.center)
        material = self.materials[0]

        return hc.Intersection(t, normal, intersection_point, material)

class Plane(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], point: glm.vec3, normal: glm.vec3):
        super().__init__(name, gtype, materials)
        self.point = point
        self.normal = glm.normalize(normal)

    def intersect(self, ray: hc.Ray):
        # TODO: Create intersect code for Plane

        # Ray-plane intersection
        denom = glm.dot(self.normal, ray.direction)
         # Check if the ray is not parallel to the plane (choosed a small value to avoid floating point errors)
        if abs(denom) > 1e-6:
            # Calculate for t
            t = glm.dot(self.point - ray.origin, self.normal) / denom
            if t > 0:
                intersection_point = ray.origin + t * ray.direction

                # Now handle checkerboard pattern if two materials are defined
                if len(self.materials) == 2:
                    # Apply checkerboard pattern based on intersection coordinates
                    checker_x = int(glm.floor(intersection_point.x))
                    checker_z = int(glm.floor(intersection_point.z))

                    # Checkerboard pattern: alternating materials based on x, z parity
                    if (checker_x + checker_z) % 2 == 0:
                        material = self.materials[0]  # First material
                    else:
                        material = self.materials[1]  # Second material
                else:
                    # If only one material is provided, just use that one
                    material = self.materials[0]
                
                return hc.Intersection(t, self.normal, intersection_point, material)
        # return None
        return hc.Intersection.default()  # No valid intersection
        # pass


class AABB(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], minpos: glm.vec3, maxpos: glm.vec3):
        super().__init__(name, gtype, materials)
        self.minpos = minpos
        self.maxpos = maxpos
  
    def intersect(self, ray: hc.Ray):
        # TODO: Create intersect code for Cube

        # Ray-box intersection using SLAB method
        # Get the necessary parameters of the intersection
        tmin = (self.minpos - ray.origin) / ray.direction
        tmax = (self.maxpos - ray.origin) / ray.direction

        t1 = glm.vec3(min(tmin.x, tmax.x), min(tmin.y, tmax.y), min(tmin.z, tmax.z))
        t2 = glm.vec3(max(tmin.x, tmax.x), max(tmin.y, tmax.y), max(tmin.z, tmax.z))

        t_near = max(t1.x, t1.y, t1.z)
        t_far = min(t2.x, t2.y, t2.z)

        # No intersection if t_near is greater than t_far or t_far is negative
        if t_near > t_far or t_far < 0:
            return None  

        # Choose the nearest intersection
        t = t_near if t_near > 0 else t_far 
        intersection_point = ray.getPoint(t)

        # Default normal for initialization
        normal = glm.vec3(0.0) 
        # Find which face the intersection occurs on
        if abs(intersection_point.x - self.minpos.x) < 1e-4: normal = glm.vec3(-1, 0, 0)
        elif abs(intersection_point.x - self.maxpos.x) < 1e-4: normal = glm.vec3(1, 0, 0)
        elif abs(intersection_point.y - self.minpos.y) < 1e-4: normal = glm.vec3(0, -1, 0)
        elif abs(intersection_point.y - self.maxpos.y) < 1e-4: normal = glm.vec3(0, 1, 0)
        elif abs(intersection_point.z - self.minpos.z) < 1e-4: normal = glm.vec3(0, 0, -1)
        elif abs(intersection_point.z - self.maxpos.z) < 1e-4: normal = glm.vec3(0, 0, 1)

        material = self.materials[0]

        return hc.Intersection(t, normal, intersection_point, material)
 

class Mesh(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], translate: glm.vec3, scale: float,
                 filepath: str):
        super().__init__(name, gtype, materials)
        verts, _, norms, self.faces, _, _ = igl.read_obj(filepath)
        self.verts = []
        self.norms = []
        for v in verts:
            self.verts.append((glm.vec3(v[0], v[1], v[2]) + translate) * scale)
        for n in norms:
            self.norms.append(glm.vec3(n[0], n[1], n[2]))

    def intersect(self, ray: hc.Ray):
        # TODO: Create intersect code for Mesh

        # Initialize with default (t = inf)
        closest_intersection = hc.Intersection.default()  

        # Iterate over each face in the mesh (where each face is a triangle)
        for face in self.faces:
            v0 = self.verts[face[0]]
            v1 = self.verts[face[1]]
            v2 = self.verts[face[2]]

            # Compute the normal of the triangle using cross product
            edge1 = v1 - v0
            edge2 = v2 - v0
            triangle_normal = glm.cross(edge1, edge2)
            triangle_normal = glm.normalize(triangle_normal)

            # Moller–Trumbore ray-triangle intersection equation
            h = glm.cross(ray.direction, edge2)
            a = glm.dot(edge1, h)

              # If the determinant is close to zero, the ray is parallel to the triangle
            if abs(a) < 1e-6: # 1e-8
                continue

            f = 1.0 / a
            s = ray.origin - v0
            u = f * glm.dot(s, h)

            # Check if the intersection point is inside or outside the triangle
            if u < 0.0 or u > 1.0:
                continue 

            q = glm.cross(s, edge1)
            v = f * glm.dot(ray.direction, q)

            # Check if the intersection point is inside or outside the triangle
            if v < 0.0 or u + v > 1.0:
                continue 

            t = f * glm.dot(edge2, q)

            # Intersection is valid and closer than previous ones
            # Configure the intersection parameters to return
            if t > 1e-6 and t < closest_intersection.t:  #1e-8
                closest_intersection.t = t
                closest_intersection.position = ray.origin + ray.direction * t
                closest_intersection.normal = triangle_normal
                closest_intersection.mat = self.materials[0]

        return closest_intersection


class Node(Geometry):
    def __init__(self, name: str, gtype: str, M: glm.mat4, materials: list[hc.Material]):
        super().__init__(name, gtype, materials)
        self.children: list[Geometry] = []
        self.M = M
        self.Minv = glm.inverse(M)

    def intersect(self, ray: hc.Ray):
        # TODO: Create intersect code for Node
        
        # Transform the ray to the local space of the node
        transformed_origin = glm.vec3(self.Minv * glm.vec4(ray.origin, 1.0))
        transformed_direction = glm.vec3(self.Minv * glm.vec4(ray.direction, 0.0))
        local_ray = hc.Ray(transformed_origin, transformed_direction)

        closest_intersection = None

        # Check intersections for each child
        for child in self.children:
            intersection = child.intersect(local_ray)
            if intersection:
                # Keep track of the closest intersection
                if not closest_intersection or intersection.t < closest_intersection.t:
                    closest_intersection = intersection

        # If we have a valid intersection
        if closest_intersection:
            # Transform the intersection point and normal back to world space
            intersection_point_world = glm.vec3(self.M * glm.vec4(closest_intersection.position, 1.0))
            normal_world = glm.normalize(glm.vec3((self.M) * glm.vec4(closest_intersection.normal, 0.0)))

            # Update the intersection result with the transformed values
            closest_intersection.position = intersection_point_world
            closest_intersection.normal = normal_world

            # If intersection material is null, use the node's material
            if not closest_intersection.mat and self.materials:
                closest_intersection.mat = self.materials[0]

        return closest_intersection





# Acceleration techniques using BVH for large meshes implementation
class LargeMeshes(Geometry):
    def __init__(self, name: str, gtype: str, materials: list, position: glm.vec3, scale: float, obj_file_path: str):
        super().__init__(name, gtype, materials)

        self.position = position
        self.scale = scale
        self.vertices = []
        self.normals = []
        self.triangles = []
        self.indices = []

        # Parse the .obj file
        self.parse_obj_file(obj_file_path)

        # Create triangles
        self.create_triangles()

        # Build BVH
        self.bvh_root = self.build_bvh(self.triangles)

    def parse_obj_file(self, file_path):
        # Parse the OBJ file to extract vertices, normals, and face indices (same as the original mesh code above)
        verts, _, norms, faces, _, _ = igl.read_obj(file_path)
        # Raise error to check if the vertices exist or not
        if verts.size == 0:
            raise ValueError(f"No vertices found in {file_path}")

        # Apply translation and scaling
        self.vertices = [(glm.vec3(v[0], v[1], v[2]) + self.position) * self.scale for v in verts]
        # Handling normals
        if norms.size > 0:
            self.normals = [glm.vec3(n[0], n[1], n[2]) for n in norms]

        self.indices = faces

    # Create the triangles from the vertex indices
    def create_triangles(self):
        for face in self.indices:
            # Triangulate the faces
            for i in range(1, len(face) - 1):
                v0 = self.vertices[face[0]]
                v1 = self.vertices[face[i]]
                v2 = self.vertices[face[i + 1]]
                normal = None
                if self.normals:
                    normal = self.normals[face[0]]
                self.triangles.append((v0, v1, v2, normal))

    # Build the BVH tree, use recursion. 
    # Here, I used SAH (Surface Area Heuristic) to split the triangles and chose the split that minimizes the cost
    def build_bvh(self, triangles):
        if len(triangles) <= 2:
            return BVHNode(self.create_bounding_box(triangles), triangles=triangles)

        # Split triangles along their centroid's dominant axis
        centroids = [((v0 + v1 + v2) / 3.0) for v0, v1, v2, _ in triangles]
        axis = max(range(3), key=lambda i: max(c[i] for c in centroids) - min(c[i] for c in centroids))
        triangles.sort(key=lambda t: ((t[0][axis] + t[1][axis] + t[2][axis]) / 3.0))

        mid = len(triangles) // 2
        left_node = self.build_bvh(triangles[:mid])
        right_node = self.build_bvh(triangles[mid:])
        bounding_box = self.create_bounding_box(triangles)
        return BVHNode(bounding_box, left=left_node, right=right_node)

    # Create the bounding box for a set of triangles
    def create_bounding_box(self, triangles):
        min_point = glm.vec3(float('inf'))
        max_point = glm.vec3(float('-inf'))

        for v0, v1, v2, _ in triangles:
            min_point = glm.min(min_point, glm.min(v0, glm.min(v1, v2)))
            max_point = glm.max(max_point, glm.max(v0, glm.max(v1, v2)))

        return BoundingBoxes(min_point, max_point, self.materials)

    # Ray-triangle intersection using the Moller–Trumbore algorithm
    def intersect(self, ray):
        return self.intersect_node(ray, self.bvh_root)

    # Check for intersection for a ray and the BVH node
    def intersect_node(self, ray, node):
        if not node.bounding_box.intersect(ray):
            return None

        if node.triangles:
            closest_intersection = None
            for triangle in node.triangles:
                intersection = self.ray_intersect_triangle(ray, triangle)
                if intersection and (closest_intersection is None or intersection.t < closest_intersection.t):
                    closest_intersection = intersection
            return closest_intersection

        left_intersection = self.intersect_node(ray, node.left)
        right_intersection = self.intersect_node(ray, node.right)

        if left_intersection and right_intersection:
            return min(left_intersection, right_intersection, key=lambda x: x.t)
        return left_intersection or right_intersection

    # Ray-triangle intersection using the Moller–Trumbore algorithm
    # Pretty much the same logic used for the Mesh class above
    def ray_intersect_triangle(self, ray, triangle):
        v0, v1, v2, _ = triangle
        epsilon = 1e-6

        edge1 = v1 - v0
        edge2 = v2 - v0
        h = glm.cross(ray.direction, edge2)
        a = glm.dot(edge1, h)

        if abs(a) < epsilon:
            return None

        f = 1.0 / a
        s = ray.origin - v0
        u = f * glm.dot(s, h)

        if u < 0.0 or u > 1.0:
            return None

        q = glm.cross(s, edge1)
        v = f * glm.dot(ray.direction, q)

        if v < 0.0 or u + v > 1.0:
            return None

        t = f * glm.dot(edge2, q)
        if t > epsilon:
            intersection_point = ray.origin + ray.direction * t
            normal = glm.normalize(glm.cross(edge1, edge2))
            material = self.materials[0]
            return hc.Intersection(t, normal, intersection_point, material)
        return None

# Node class for the BVH
class BVHNode:
    def __init__(self, bounding_box, left=None, right=None, triangles=None):
        self.bounding_box = bounding_box
        self.left = left
        self.right = right
        self.triangles = triangles

# AABB class for the bounding boxes used for the BVH
class BoundingBoxes:
    def __init__(self, min_point, max_point, materials):
        self.min_point = min_point
        self.max_point = max_point
        self.materials = materials

    # Check for intersection with the ray and the bounding box
    # Pretty much the same logic used for the AABB class above
    def intersect(self, ray):
        tmin = (self.min_point - ray.origin) / ray.direction
        tmax = (self.max_point - ray.origin) / ray.direction

        t1 = glm.vec3(min(tmin.x, tmax.x), min(tmin.y, tmax.y), min(tmin.z, tmax.z))
        t2 = glm.vec3(max(tmin.x, tmax.x), max(tmin.y, tmax.y), max(tmin.z, tmax.z))

        t_near = max(t1.x, t1.y, t1.z)
        t_far = min(t2.x, t2.y, t2.z)

        if t_near > t_far or t_far < 0:
            return None

        t = t_near if t_near > 0 else t_far
        intersection_point = ray.getPoint(t)

        normal = glm.vec3(0.0)
        if abs(intersection_point.x - self.min_point.x) < 1e-4: normal = glm.vec3(-1, 0, 0)
        elif abs(intersection_point.x - self.max_point.x) < 1e-4: normal = glm.vec3(1, 0, 0)
        elif abs(intersection_point.y - self.min_point.y) < 1e-4: normal = glm.vec3(0, -1, 0)
        elif abs(intersection_point.y - self.max_point.y) < 1e-4: normal = glm.vec3(0, 1, 0)
        elif abs(intersection_point.z - self.min_point.z) < 1e-4: normal = glm.vec3(0, 0, -1)
        elif abs(intersection_point.z - self.max_point.z) < 1e-4: normal = glm.vec3(0, 0, 1)

        return hc.Intersection(t, normal, intersection_point, self.materials[0])


