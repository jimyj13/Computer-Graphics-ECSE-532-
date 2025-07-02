# Yong-Jung Song 260735417
import numpy as np

# Nones initially and updated in HalfedgeMesh.build(), since we only have the vertex positionsm and face vertex indices
class Vertex:
    def __init__(self, point):
        self.point = point
        self.halfedge = None
        self.index = None

class Halfedge:
    def __init__(self):
        self.vertex = None # source vertex
        self.twin = None
        self.next = None
        self.prev = None
        self.edge = None
        self.face = None
        self.index = None

class Edge:
    def __init__(self):
        self.halfedge = None
        self.index = None

class Face:
    def __init__(self, vertices):
        self.vertices = vertices
        self.halfedge = None
        self.index = None

class HalfedgeMesh:
    def __init__(self, vertices, faces):
        self.vertices = np.array([Vertex(pos) for pos in vertices])
        self.halfedges = []
        self.edges = []
        self.faces = []
        for face_vertex_ids in faces:
            face_vertices = [self.vertices[id] for id in face_vertex_ids]
            self.faces.append(Face(face_vertices))
        self.build()

    # Convenience functions to create new elements
    def new_vertex(self, point):
        vertex = Vertex(point)
        vertex.index = len(self.vertices)
        self.vertices = np.append(self.vertices, vertex)
        return vertex

    def new_face(self, vertices):
        face = Face(vertices)
        face.index = len(self.faces)
        self.faces = np.append(self.faces, face)
        return face

    def new_edge(self):
        edge = Edge()
        edge.index = len(self.edges)
        self.edges = np.append(self.edges, edge)
        return edge

    def new_halfedge(self):
        he = Halfedge()
        he.index = len(self.halfedges)
        self.halfedges = np.append(self.halfedges, he)
        return he
    
    '''
    Given HalfedgeMesh object (potentially with quads or ngons), return a tuple of numpy arrays (vertices, triangles, triangle_to_face) for rendering.
    vertices: (n, 3) array of vertex positions
    triangles: (m, 3) array of vertex indices forming triangles
    triangle_to_face: (m,) array of face indices corresponding to each triangle (needed for face selection especially) [tri_index] -> face_index
    '''
    def get_vertices_and_triangles(self):
        vertices = [vertex.point for vertex in self.vertices]
        triangles = []
        triangle_to_face = [] # map from triangle to face, {to generalize to n-gons}
        i = 0
        for face in self.faces:
            if len(face.vertices) == 3:
                triangles.append([vertex.index for vertex in face.vertices])
                triangle_to_face.append(i)
            else:
                # implement simple ear clipping algorithm
                triangles_vertices = triangulate([vertex for vertex in face.vertices])
                for triangle_vertices_triple in triangles_vertices:
                    triangles.append(triangle_vertices_triple)
                    triangle_to_face.append(i)
            i += 1
        return np.array(vertices), np.array(triangles), triangle_to_face

    # Build the halfedge data structure from the vertex positions and face vertex indices stored in self.vertices and self.faces
    # This is essential for all following objectives to work
    def build(self):
        # TODO: Objective 1: build the halfedge data structure
        # Hint: use a dict to keep track of edges, as halfedges are being created
        
        # Assign indices to vertices and faces using enumerate for clarity
        for index, vertex in enumerate(self.vertices):
            vertex.index = index
        for index, face in enumerate(self.faces):
            face.index = index

        # Dictionary to keep track of edges
        edges_dict = {}
        
        # Function to create a halfedge associated with a vertex
        def create_halfedge_for_vertex(vertex: Vertex):
            halfedge = self.new_halfedge()
            halfedge.vertex = vertex
            vertex.halfedge = halfedge
            return halfedge

        # Function to handle edge creation and connection of twin halfedges
        def connect_halfedge_and_edge(halfedge, start_vertex: Vertex, end_vertex: Vertex):
            edge_key = (end_vertex, start_vertex)
            if edge_key in edges_dict:  # Check for an existing twin
                existing_edge = edges_dict[edge_key]
                twin_halfedge = existing_edge.halfedge
                halfedge.twin = twin_halfedge
                twin_halfedge.twin = halfedge
                halfedge.edge = existing_edge
            else:
                new_edge = self.new_edge()
                new_edge.halfedge = halfedge
                halfedge.edge = new_edge
                edges_dict[(start_vertex, end_vertex)] = new_edge

        # Iterate through each face to create halfedges
        for face in self.faces:
            halfedge_list = [create_halfedge_for_vertex(vertex) for vertex in face.vertices]

            for i in range(len(halfedge_list)):
                current_halfedge = halfedge_list[i]
                next_halfedge = halfedge_list[(i + 1) % len(halfedge_list)]
                current_halfedge.next = next_halfedge
                current_halfedge.prev = halfedge_list[(i - 1) % len(halfedge_list)]
                current_halfedge.face = face

            face.halfedge = halfedge_list[0]

            # Connect edges for each halfedge
            for i in range(len(halfedge_list)):
                connect_halfedge_and_edge(halfedge_list[i], face.vertices[i], face.vertices[(i + 1) % len(halfedge_list)])

        self.sanity_check()
        pass

    # Given a face, loop over its halfedges he in order to update face.vertices and he.face after some operation
    def update_he_vertices_around_face(self, face):
        he = face.halfedge
        vertices = []
        while True:
            vertices.append(he.vertex)
            he = he.next
            he.face = face # update he face
            if he.index == face.halfedge.index:
                break
        face.vertices = vertices # update face vertices

    # Given an edge, with both sides being triangles, flip the edge
    #           v1                              v1
    #           /\                              /\
    #          /  \                            / |\
    #         /    \                          /  | \
    #        /______\                        /   |  \
    #          edge      -> flip edge ->         |
    #        \      /                        \   |  /
    #         \    /                          \  | /
    #          \  /                            \ |/
    #           \/                              \/

    def flip_edge(self, edge):
        # TODO: Objective 3a: flip the edge (only if both sides are triangles)
        # Configure target halfedges 1 and 2 and faces 1 and 2
        he1 = edge.halfedge
        he2 = he1.twin
        face1 = he1.face
        face2 = he2.face

        # Apply the flip edges only when the adjacent faces are triangles
        if (face1 and face2 and len(face1.vertices) == 3 and len(face2.vertices) == 3):

            # Store the next halfedge of he1 for later use
            next_he1 = he1.next.next
            next_he2 = he2.next.next

            # Function to update the halfedge's vertex and connections
            def update_halfedge(he, new_next):
                vertex = he.prev.vertex
                halfedge = he
                prev_halfedge = he.next
                next_halfedge = new_next
                return vertex, halfedge, prev_halfedge, next_halfedge

            # Update halfedge he1's vertex and connections
            old_vertex_he1 = he1.vertex
            he1.vertex, he1.vertex.halfedge, he1.prev, he1.next = update_halfedge(he1, next_he2)
            # print(f"Updated he1: vertex {old_vertex_he1.index} to vertex {he1.vertex.index}")

            # Update halfedge he2's vertex and connections
            old_vertex_he2 = he2.vertex
            he2.vertex, he2.vertex.halfedge, he2.prev, he2.next = update_halfedge(he2, next_he1)
            # print(f"Updated he2: vertex {old_vertex_he2.index} to vertex {he2.vertex.index}")

            # Connect halfedges he1 and he2
            halfedges = (he1, he2)
            for halfedge in halfedges:
                he_a = halfedge.next
                he_b = halfedge.prev
                he_a.next = he_b
                he_a.prev = halfedge
                he_b.next = halfedge
                he_b.prev = he_a

            # Assign the new halfedges to their respective faces
            face1.halfedge = he1
            face2.halfedge = he2

            # Update the vertices around each face
            self.update_he_vertices_around_face(face1)
            self.update_he_vertices_around_face(face2)
            self.update_indices()
        # If the adjacent faces are not triangles, print an error message    
        else:
            print("Edge cannot be flipped; both adjacent faces must be triangles.")
            return
        self.sanity_check()


    # Given an edge, with both sides being triangles, split the edge in the middle, creating a new vertex and connecting
    # it to the facing corners of the 2 triangles
    #           v1                              v1
    #           /\                              /\
    #          /  \                            / |\
    #         /    \                          /  | \
    #        /______\                        /   |  \
    #          edge      -> split edge ->    ---v2---
    #        \      /                        \   |  /
    #         \    /                          \  | /
    #          \  /                            \ |/
    #           \/                              \/

    def split_edge(self, edge):
            # TODO: Objective 3b: split the edge (only if both sides are triangles)
            # Get information about the selected edge and the two adjacent faces
            he1 = edge.halfedge
            he2 = he1.twin
            face1 = he1.face
            face2 = he2.face
            # Only apply splits if the two adjacent faces are triangles
            if (face1 and face2 and len(face1.vertices) == 3 and len(face2.vertices) == 3):
                he1.face.halfedge = he1
                he2.face.halfedge = he2
                v3 = he1.prev.vertex
                v4 = he2.prev.vertex

                # Create the midpoint vertex
                v1 = he1.vertex
                v2 = he2.vertex
                midpoint = (v2.point + v1.point) / 2
                midpoint = self.new_vertex(midpoint)
                midpoint.halfedge = he1.twin

                # Configure halfedge, edge, and face pointers for face1
                he1_a = he1.next
                he1_b = he1.prev
                v3 = he1_b.vertex
                he1_edge = he1.edge

                # Configure halfedge, edge, and face pointers for face2
                he2_a = he2.next
                he2_b = he2.prev
                v4 = he2_b.vertex

                # Split face1 into two triangles so we can get four triangles in the end
                he1_sp = self.new_halfedge()
                he1_c = self.new_halfedge()
                he1_d = self.new_halfedge()
                he1_sp_edge = self.new_edge()
                he1_c_edge = self.new_edge()
                he1_sp_edge.halfedge = he1_sp
                he1_c_edge.halfedge = he1_c
                face1_c = self.new_face(None)
                face1_d = face1
                # Define the parameters for each set of halfedge updates in a list of dictionaries
                halfedge_params_face1x_part1 = [
                    {'halfedge': he1_sp, 'vertex': midpoint, 'twin': he2, 'next': he1_a, 'previous': he1_c, 'edge': he1_sp_edge, 'face': face1_c},
                    {'halfedge': he1_c, 'vertex': v3, 'twin': he1_d, 'next': he1_sp, 'previous': he1_a, 'edge': he1_c_edge, 'face': face1_c}
                ]
                halfedge_params_face1x_part2 = [
                    {'halfedge': he1_a, 'vertex': v2, 'next': he1_c, 'previous': he1_sp, 'face': face1_c}
                ]

                # Loop through each set of parameters and directly update the halfedge properties
                for params in halfedge_params_face1x_part1:
                    he = params['halfedge']
                    if 'vertex' in params: he.vertex = params['vertex']
                    if 'twin' in params: he.twin = params['twin']
                    if 'next' in params: he.next = params['next']
                    if 'previous' in params: he.prev = params['previous']
                    if 'edge' in params: he.edge = params['edge']
                    if 'face' in params: he.face = params['face']

                # Loop through the parameters and directly update the halfedge properties
                for params in halfedge_params_face1x_part2:
                    he = params['halfedge']
                    if 'vertex' in params: he.vertex = params['vertex']
                    if 'next' in params: he.next = params['next']
                    if 'previous' in params: he.prev = params['previous']
                    if 'face' in params: he.face = params['face']
                face1_c.halfedge = he1_c

                # Define face1_d:
                # First configure the halfedge
                he2_sp = self.new_halfedge()
                # Define the parameters for both halfedge updates in a list of dictionaries
                halfedge_params_face1y_part1 = [
                    {'halfedge': he1, 'vertex': v1, 'twin': he2_sp, 'next': he1_d, 'previous': he1_b, 'edge': he1_edge, 'face': face1_d},
                    {'halfedge': he1_d, 'vertex': midpoint, 'twin': he1_c, 'next': he1_b, 'previous': he1, 'edge': he1_c_edge, 'face': face1_d}
                ]
                halfedge_params_face1y_part2 = [
                    {'halfedge': he1_b, 'vertex': v3, 'next': he1, 'previous': he1_d, 'face': face1_d}
                ]

                # Loop through each dictionary of parameters and update the halfedge properties
                for params in halfedge_params_face1y_part1:
                    he = params['halfedge']
                    if 'vertex' in params: he.vertex = params['vertex']
                    if 'twin' in params: he.twin = params['twin']
                    if 'next' in params: he.next = params['next']
                    if 'previous' in params: he.prev = params['previous']
                    if 'edge' in params: he.edge = params['edge']
                    if 'face' in params: he.face = params['face']

                # Loop through the dictionary and update the halfedge properties
                for params in halfedge_params_face1y_part2:
                    he = params['halfedge']
                    if 'vertex' in params: he.vertex = params['vertex']
                    if 'next' in params: he.next = params['next']
                    if 'previous' in params: he.prev = params['previous']
                    if 'facae' in params: he.face = params['face']
                face1_d.halfedge = he1_d

                # Split face2 into two triangles so we can get four triangles in the end
                he2_c = self.new_halfedge()
                he2_d = self.new_halfedge()
                he2_c_edge = self.new_edge()
                he2_c_edge.halfedge = he2_c
                face2_c = self.new_face(None)
                face2_d = face2

                # Define face2_c:
                # Define the parameters for both halfedge updates in a list of dictionaries
                halfedge_params_face2x_part1 = [
                    {'halfedge': he2_sp, 'vertex': midpoint, 'twin': he1, 'next': he2_a, 'previous': he2_c, 'edge': he1_edge, 'face': face2_c},
                    {'halfedge': he2_c, 'vertex': v4, 'twin': he2_d, 'next': he2_sp, 'previous': he2_a, 'edge': he2_c_edge, 'face': face2_c}
                ]

                halfedge_params_face2x_part2 = [
                    {'halfedge': he2_a, 'vertex': v1, 'next': he2_c, 'previous': he2_sp, 'face': face2_c}
                ]

                # Loop through each dictionary of parameters and update the halfedge properties
                for params in halfedge_params_face2x_part1:
                    he = params['halfedge']
                    if 'vertex' in params: he.vertex = params['vertex']
                    if 'twin' in params: he.twin = params['twin']
                    if 'next' in params: he.next = params['next']
                    if 'previous' in params: he.prev = params['previous']
                    if 'edge' in params: he.edge = params['edge']
                    if 'face' in params: he.face = params['face']

                # Loop through the dictionary and update the halfedge properties
                for params in halfedge_params_face2x_part2:
                    he = params['halfedge']
                    if 'vertex' in params: he.vertex = params['vertex']
                    if 'next' in params: he.next = params['next']
                    if 'previous' in params: he.prev = params['previous']
                    if 'face' in params: he.face = params['face']
                face2_c.halfedge = he2_c

                # Define face2_d:
                # Define the parameters for both halfedge updates in a list of dictionaries
                halfedge_params_face2y_part1 = [
                    {'halfedge': he2, 'vertex': v2, 'twin': he1_sp, 'next': he2_d, 'previous': he2_b, 'edge': he1_sp_edge, 'face': face2_d},
                    {'halfedge': he2_d, 'vertex': midpoint, 'twin': he2_c, 'next': he2_b, 'previous': he2, 'edge': he2_c_edge, 'face': face2_d}
                ]

                halfedge_params_face2y_part2 = [
                    {'halfedge': he2_b, 'vertex': v4, 'next': he2, 'previous': he2_d, 'face': face2_d}
                ]

                # Loop through each dictionary of parameters and update the halfedge properties
                for params in halfedge_params_face2y_part1:
                    he = params['halfedge']
                    if 'vertex' in params: he.vertex = params['vertex']
                    if 'twin' in params: he.twin = params['twin']
                    if 'next' in params: he.next = params['next']
                    if 'previous' in params: he.prev = params['previous']
                    if 'edge' in params: he.edge = params['edge']
                    if 'face' in params: he.face = params['face']

                # Loop through the dictionary and update the halfedge properties
                for params in halfedge_params_face2y_part2:
                    he = params['halfedge']
                    if 'vertex' in params: he.vertex = params['vertex']
                    if 'next' in params: he.next = params['next']
                    if 'previous' in params: he.prev = params['previous']
                    if 'face' in params: he.face = params['face']
                face2_d.halfedge = he2_d

                # Set up face vertices:
                for face in [face1_c, face1_d, face2_c, face2_d]:
                    self.update_he_vertices_around_face(face)    
            # If the adjacent faces are not triangles, print an error message                
            else:
                print("Edge cannot be split; both adjacent faces must be triangles.")
                return

            self.sanity_check()

    # Given an edge, dissolve (erase) the edge, merging the 2 faces on its sides
    def erase_edge(self, edge):
        # TODO: Objective 3c: erase the edge
        # Get the halfedges of the selected edge
        he1 = edge.halfedge
        he2 = he1.twin

        # Get the two adjacent faces of the selected edge
        face1 = he1.face
        face2 = he2.face

        # Remove the edge from the mesh
        self.edges = [e for e in self.edges if e != edge]
        # Update the indices after the deletion process
        self.update_indices() 

        # Update the halfedges' links for both faces
        he1_prev = he1.prev
        he2_prev = he2.prev

        he1_prev.next = he2.next
        he2_prev.next = he1.next

        he1.next.prev = he2_prev
        he2.next.prev = he1_prev

        # Update the face's halfedges
        face1.halfedge = he1_prev.next

        #Remove face2 from the list of faces
        self.faces = [f for f in self.faces if f != face2]  #use list to avoid np.array issue

        # Remove the halfedges from the list of halfedges
        self.halfedges = [he for he in self.halfedges if he not in (he1, he2)]  # Ensure references remain intact
        self.update_indices() 
        # Update the halfedge vertices of each face
        self.update_he_vertices_around_face(face1)

        #update indices after the deletion process
        self.update_indices() 
        self.sanity_check()


    # Helper function made to iterate over halfedges around a given face (used for extrude_face_topological)    
    def iterate_halfedges_around_face(self, face):
        start_he = face.halfedge  # Start from the face's halfedge
        he = start_he
        while True:
            yield he
            he = he.next  # Move to the next halfedge in the loop
            if he == start_he:  # Stop when we've looped back to the start
                break

    # Since extrude_face, inset_face, bevel_face, all update connectivity in the exact same way, implement this topological
    # operation as a separate function (no need to call it inside the other functions, that's already done)
    def extrude_face_topological(self, face):
        # TODO: Objective 4a: implement the extrude operation,
        # Note that for each edge of the face, you need to create a new face (and accordingly new halfedges, edges, vertices)
        # Hint: Count the number of elements you need to create per each new face, Creating them all before updating connectivity may make it easier
      
        # Iterate through the selected face and get the original halfedges and vertices
        original_halfedges = list(self.iterate_halfedges_around_face(face))
        original_vertices = [he.vertex for he in original_halfedges]

        # Duplicate the vertices for the new, extruded face
        new_vertices = [self.new_vertex(np.array(v.point)) for v in original_vertices]

        # Create new halfedges, edges, and connect new quad faces for extrusion
        new_halfedges = []  # Store new halfedges for the extruded face
        for i, he in enumerate(original_halfedges):
            # Create new halfedges for the extruded face
            new_he = self.new_halfedge()
            new_he.vertex = new_vertices[i]
            new_he.vertex.halfedge = new_he
            new_halfedges.append(new_he)

            # Create the quad face for extrusion
            next_he = original_halfedges[(i + 1) % len(original_halfedges)]
            quad_face = self.new_face([he.vertex, next_he.vertex, new_vertices[(i + 1) % len(new_vertices)], new_vertices[i]])

            # Create four halfedges for the quad face
            he1 = original_halfedges[i]  # first halfedge of a quad should be the halfedge from the original face
            he2 = self.new_halfedge()
            he3 = self.new_halfedge()  
            he4 = self.new_halfedge()  

            # Link the halfedges in a loop
            he1.next, he2.next, he3.next, he4.next = he2, he3, he4, he1
            he1.prev, he2.prev, he3.prev, he4.prev = he4, he1, he2, he3

            he3.twin = new_he
            new_he.twin = he3

            # Assign vertices to halfedges for connection
            he1.vertex = he.vertex
            he4.vertex = new_vertices[i]
            he3.vertex = new_vertices[(i + 1) % len(new_vertices)]
            he2.vertex = next_he.vertex

            # Set the face for each halfedge
            for quad_he in [he1, he2, he3, he4]:
                quad_he.face = quad_face

            # Set halfedges on new quad face and vertices
            quad_face.halfedge = he1
            he.vertex.halfedge = he1
            next_he.vertex.halfedge = he4

        # Link (make connections for) the quad faces
        for i, old_he in enumerate(original_halfedges):
            he1 = original_halfedges[i]
            he2 = original_halfedges[ (i+1+len(original_halfedges))%len(original_halfedges) ]
            he1_next = he1.next
            he2_prev = he2.prev
            edge = self.new_edge()
            edge.halfedge = he1_next
            he1_next.twin = he2_prev
            he2_prev.twin = he1_next
            he1_next.edge = edge
            he2_prev.edge = edge

        # Link the new halfedges in a loop
        for i, new_he in enumerate(new_halfedges):
            new_he.next = new_halfedges[(i + 1) % len(new_halfedges)]
            new_he.prev = new_halfedges[(i - 1) % len(new_halfedges)]
            # NEW:
            edge = self.new_edge()
            edge.halfedge = new_he
            new_he.edge = edge
            he_b = original_halfedges[i].next.next
            he_b.edge = edge
            new_he.twin = he_b

        # Update the connectivity for the original face
        face.halfedge = new_halfedges[0]

        # Update vertices and halfedges for the original and new faces
        self.update_he_vertices_around_face(face)  # Update original face
        for quad_face in self.faces[-len(original_halfedges):]:  # Update the new quad faces created for extrusion
            self.update_he_vertices_around_face(quad_face)

        # Indices update just in case and sanity check
        self.update_indices()
        self.sanity_check()



    def inset_face(self, face, t): # t=0, no inset, t=1, full inset (face shrinks to barycenter)
        # TODO: Objective 4b: implement the inset operation,
        # Scale the face towards the barycenter based on the parameter t
        # This is pretty much the exact same thing like scaling but just wanted to write it out
        
        # Calculate the barycenter of the face
        barycenter = np.mean([vertex.point for vertex in face.vertices], axis=0)
        
        # Inset the face towards the barycenter by t
        for vertex in face.vertices:
            vertex.point = vertex.point * (1 - t) + barycenter * t
        self.sanity_check()
        pass


    def extrude_face(self, face, t): # t=0, no extrusion, t<0, inwards, t>0 outwards
        # TODO: Objective 4b: implement the extrude operation,
        # Get the normal of the face
        face_vertices = [vertex for vertex in face.vertices]
        normal = np.cross(face_vertices[1].point - face_vertices[0].point,
                        face_vertices[2].point - face_vertices[0].point)
        normal /= np.linalg.norm(normal)  # Normalize the normal vector


        # Move the vertices of the extruded face along the normal direction
        for vertex in face_vertices:
            vertex.point += normal * t
        self.sanity_check()
        pass

    def bevel_face(self, face, tx, ty): # ty for the normal extrusion, tx for the scaling (tangentially)
        #CHECK IF I SHOULD INSET THEN EXTRUDE OR VICE VERSA
        
        # First, inset the face by tx
        self.inset_face(face, tx)

        # Then, extrude the inset face along its normal by ty
        self.extrude_face(face, ty)
        pass

    def scale_face(self, face, t): # t=0, no inset, t=1, full inset (face shrinks to barycenter)
        barycenter = np.mean([vertex.point for vertex in face.vertices], axis=0)
        for vertex in face.vertices:
            vertex.point = vertex.point * (1 - t) + barycenter * t

    # need to update HalfedgeMesh indices after deleting elements
    def update_indices(self):
        for i, vertex in enumerate(self.vertices):
            vertex.index = i
        for i, face in enumerate(self.faces):
            face.index = i
        for i, edge in enumerate(self.edges):
            edge.index = i
        for i, he in enumerate(self.halfedges):
            he.index = i

    # Helpful for debugging after each local operation
    def sanity_check(self):
        for i,f in enumerate(self.faces):
            if f.halfedge is None:
                print('Face {} has no halfedge'.format(i))
            if f.index != i:
                print('Face {} has wrong index'.format(i))
            for v in f.vertices:
                if v is None:
                    print('Face {} has a None vertex'.format(i))

        for i,e in enumerate(self.edges):
            if e.halfedge is None:
                print('Edge {} has no halfedge'.format(i))
            if e.index != i:
                print('Edge {} has wrong index'.format(i))

        for i,v in enumerate(self.vertices):
            if v.halfedge is None:
                print('Vertex {} has no halfedge'.format(i))
            if v.index != i:
                print('Vertex {} has wrong index'.format(i))

        for i,he in enumerate(self.halfedges):
            if he.vertex is None:
                print('Halfedge {} has no vertex'.format(i))
            if he.index != i:
                print('Halfedge {} has wrong index'.format(i))
            if he.face is None:
                print('Halfedge {} has no face'.format(i))
            if he.edge is None:
                print('Halfedge {} has no edge'.format(i))
            if he.next is None:
                print('Halfedge {} has no next'.format(i))
            if he.prev is None:
                print('Halfedge {} has no prev'.format(i))
            if he.twin is None:
                print('Halfedge {} has no twin'.format(i))

def triangulate(vertices):
    # TODO: Objective 2: implement the simple triangulation algorithm as described in the readme
    num_vertices = len(vertices)
    # Ensure there are enough vertices to form triangles
    # So if it's less than 3 vertices (cannot form a polygon), just return
    if num_vertices < 3:
        # A polygon with fewer than 3 vertices is not valid for triangulation
        print("Cannot triangulate a polygon with fewer than 3 vertices.")
        return []
    
    # Store list of triangles
    triangles_list = []

    # Create the initial triangle using the first three vertices
    initial_triangle = list([vertices[0].index, vertices[1].index, vertices[2].index])
    triangles_list.append(initial_triangle)

    # Create triangles for the rest of the vertices
    for i in range(3, num_vertices):
        triangle = [
            vertices[0].index,      # always the first vertex
            vertices[i - 1].index,  # the previous vertex
            vertices[i].index       # the current vertex
        ]
        triangles_list.append(list(triangle))  # Using list() to create a new list

    return triangles_list

    # pass

