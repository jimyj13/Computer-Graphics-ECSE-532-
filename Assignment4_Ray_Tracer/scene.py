# Yong-Jung Song 260735417

import math
import glm
import numpy as np
import geometry as geom
import helperclasses as hc
from tqdm import tqdm

import random

class Scene:

    def __init__(self,
                 width: int,
                 height: int,
                 jitter: bool,
                 samples: int,
                 eye_position: glm.vec3,
                 lookat: glm.vec3,
                 up: glm.vec3,
                 fov: float,
                 ambient: glm.vec3,
                 lights: list[hc.Light],
                 objects: list[geom.Geometry],
                 aperture_radius: float = 0.0,  # Added DOF parameters
                 focal_distance: float = 1.0,
                 path_tracing=False, 
                 area_lights=False 
                 ):
        self.width = width  # width of image
        self.height = height  # height of image
        self.aspect = width / height  # aspect ratio
        self.jitter = jitter  # should rays be jittered
        self.samples = samples  # number of rays per pixel
        self.eye_position = eye_position  # camera position in 3D
        self.lookat = lookat  # camera look at vector
        self.up = up  # camera up position
        self.fov = fov  # camera field of view
        self.ambient = ambient  # ambient lighting
        self.lights = lights  # all lights in the scene
        self.objects = objects  # all objects in the scene
        self.aperture_radius = aperture_radius  # DOF
        self.focal_distance = focal_distance  # DOF
        self.path_tracing = path_tracing  # Initialize path_tracing flag
        self.area_lights = area_lights  # Initialize area_lights flag
        
    
    #------------------
    def render(self):
        image = np.zeros((self.height, self.width, 3))   # image with row,col indices and 3 channels, origin is top left

        cam_dir = self.eye_position - self.lookat
        distance_to_plane = 1.0
        top = distance_to_plane * math.tan(0.5 * math.pi * self.fov / 180)
        right = self.aspect * top
        bottom = -top
        left = -right

        w = glm.normalize(cam_dir)
        u = glm.cross(self.up, w)
        u = glm.normalize(u)
        v = glm.cross(w, u)

        # Generate and vectorize rays
        x = np.linspace(0.5, self.width - 0.5, self.width)  # pixel column
        y = np.linspace(0.5, self.height - 0.5, self.height)  # pixel row

        X, Y = np.meshgrid(x, y)  # Create 2D grid for all pixels

        # Normalize pixel coordinates (flip y)
        norm_x = (X) / self.width
        norm_y = (Y) / self.height
        norm_y = 1.0 - norm_y  # Flip y-coordinate for correct orientation

        # Calculate pixel positions in camera space
        u_coord = left + (right - left) * norm_x
        v_coord = bottom + (top - bottom) * norm_y
        pixel_positions = self.eye_position + u_coord[:, :, np.newaxis] * u + v_coord[:, :, np.newaxis] * v - distance_to_plane * w

        # Generate rays (vectors)
        
        directions = pixel_positions - self.eye_position
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)  # Normalize the vectors

    

        # Class used for the DOF blur, in order to make the blurs more natural
        def poisson_disk_sample(aperture_radius, num_samples):
            points = []
            while len(points) < num_samples:
                x, y = random.uniform(-1, 1), random.uniform(-1, 1)
                if x ** 2 + y ** 2 <= 1:
                    points.append(glm.vec3(x * aperture_radius, y * aperture_radius, 0.0))
            return points

        

        # TODO: Generate rays
        # Function to generate rays, and return it when called.
        def generate_ray(col, row, jitter_x=0.0, jitter_y=0.0):
            # Normalize pixel coordinates and flip y
            x = (col + 0.5 + jitter_x) / self.width
            y = (row + 0.5 + jitter_y) / self.height
            y = 1.0 - y  # Flip y-coordinate for correct orientation

            # Calculate pixel position in camera space
            u_coord = left + (right - left) * x
            v_coord = bottom + (top - bottom) * y
            pixel_position = self.eye_position + u_coord * u + v_coord * v - distance_to_plane * w

            # When we configure the aperture radius for DOF, apply the DOF effect
            if self.aperture_radius > 0.0:
                # Stratified sampling on lens for DOF
                lens_offsets = poisson_disk_sample(self.aperture_radius, self.samples)
                averaged_rays = []

                for lens_offset in lens_offsets:
                    # Offset ray origin for aperture sampling
                    ray_origin = self.eye_position + lens_offset
                    # Compute focal point and ray direction
                    focal_point = self.eye_position + glm.normalize(pixel_position - self.eye_position) * self.focal_distance
                    direction = glm.normalize(focal_point - ray_origin)
                    averaged_rays.append(hc.Ray(ray_origin, direction))  # Add ray to list
                    smallest_ray = min(averaged_rays, key=lambda ray: np.linalg.norm(ray.direction))
                # find the smallest ray (distance) and return it
                return smallest_ray
            
            # If no DOF blur, do the regular ray generation
            else:
                # For no aperture, return a single ray
                ray_origin = self.eye_position
                direction = glm.normalize(pixel_position - self.eye_position)
                return hc.Ray(ray_origin, direction)
        
        # TODO: Test for intersection with all objects
        # Function to test for intersections with all objects in the scene.
        # Returns closest intersection if found, else returns None
        def test_intersections(ray):
            closest_intersection = hc.Intersection.default()
            closest_intersection.t = float('inf')

            # Iterate through the objects in the scene and get the intersection
            for obj in self.objects:
                intersection = obj.intersect(ray)

                # Handle cases where no intersection is returned
                if intersection is not None and intersection.t < closest_intersection.t:
                    closest_intersection = intersection

            # If no valid intersection was found, closest_intersection.t will remain 'inf'
            if closest_intersection.t == float('inf'):
                return None  # Return None to indicate no intersection found

            return closest_intersection
        
        # Fresnel reflectance implementation. Only is accessed when the .json file specifies a reflection index
        def fresnel_reflectance(incident, normal, reflection_index):
            cosi = glm.clamp(glm.dot(incident, normal), -1.0, 1.0)
            etai = 1.0  # Refractive index of the incident medium (air)
            etat = reflection_index  # Refractive index of the material

            # Snell's law for refraction angle (here we wanted to separate refraction so I used reflection index)
            sint = etai / etat * math.sqrt(max(0.0, 1.0 - cosi * cosi))
            if sint >= 1.0:  # Total internal reflection
                return 1.0

            cost = math.sqrt(max(0.0, 1.0 - sint * sint))
            cosi = abs(cosi)

            # Fresnel equations
            R0 = ((etai - etat) / (etai + etat)) ** 2
            r_parallel = ((etat * cosi - etai * cost) / (etat * cosi + etai * cost)) ** 2
            r_perpendicular = ((etai * cosi - etat * cost) / (etai * cosi + etat * cost)) ** 2
            return 0.5 * (r_parallel + r_perpendicular)
        
    

        # Refraction implementation, we use Snell's law for refraction angle
        def refract(ray_dir, normal, eta_ratio):
            cos_theta = min(glm.dot(-ray_dir, normal), 1.0)
            r_out_perpendicular = eta_ratio * (ray_dir + cos_theta * normal)
            r_out_parallel = -math.sqrt(abs(1.0 - glm.length2(r_out_perpendicular))) * normal
            return glm.normalize(r_out_perpendicular + r_out_parallel)

        # Ray tracing algorithm.
        def ray_trace(ray, depth=0, max_depth=1): #10

            if depth > max_depth:
                return glm.vec3(0.0, 0.0, 0.0)  # Return black if recursion exceeds max depth
            
            # Test for intersection with objects in the scene
            intersection = test_intersections(ray)
            if intersection is None:
                return glm.vec3(0.0, 0.0, 0.0)  # Background color

            # Compute color normally
            color = compute_shading(intersection, ray, depth, max_depth)
            reflectance = fresnel_reflectance(ray.direction, intersection.normal, intersection.mat.reflection_index)
            if ray is not None and depth < max_depth:

                #If we set a reflection index > 1.0, we apply Fresnel Reflection
                if intersection.mat.reflection_index > 1.0:
                    reflection_ray = hc.Ray(intersection.position + 1e-5 * intersection.normal, glm.reflect(ray.direction, intersection.normal))
                    reflection_color = ray_trace(reflection_ray, depth + 1, max_depth)
                    color += reflectance * reflection_color

                # If we set a refraction index > 1.0, we apply Refraction
                if intersection.mat.refraction_index > 1.0:
                    eta_ratio = (1.0 / intersection.mat.refraction_index 
                                if glm.dot(ray.direction, intersection.normal) < 0
                                else intersection.mat.refraction_index / 1.0)

                    normal = intersection.normal if glm.dot(ray.direction, intersection.normal) < 0 else -intersection.normal

                    refracted_dir = refract(ray.direction, normal, eta_ratio)
                    if refracted_dir is not None:
                        refraction_ray = hc.Ray(intersection.position - 1e-5 * normal, refracted_dir)
                        refraction_color = ray_trace(refraction_ray, depth + 1, max_depth)
                        color += (1.0 - reflectance) * refraction_color

            return glm.clamp(color, glm.vec3(0.0), glm.vec3(1.0))
        


        # Path tracing Implementation
         # Path tracing function (recursive path tracing method)
         # Includes Fresnel reflection, indirect lighting, and refraction
        def path_trace(ray, depth=0, max_depth=3): #5. For better quality do a max_depth of 5 but for rendering time I used 3
            if depth > max_depth:
                return glm.vec3(0.0, 0.0, 0.0)  # Terminate recursion after max_depth

            # Test for intersection with objects in the scene
            intersection = test_intersections(ray)
            if intersection is None:
                return glm.vec3(0.0, 0.0, 0.0)  # Background color (no intersection)

            # Compute shading for the intersection
            color = compute_shading(intersection, ray, depth, max_depth)

            reflectance = fresnel_reflectance(ray.direction, intersection.normal, intersection.mat.reflection_index)
            if ray is not None and depth < max_depth:
                # Fresnel Reflection
                if intersection.mat.reflection_index > 1.0:
                        reflection_dir = glm.reflect(ray.direction, intersection.normal)
                        reflection_ray = hc.Ray(intersection.position + 1e-5 * intersection.normal, reflection_dir)
                        reflection_color = path_trace(reflection_ray, depth + 1, max_depth)
                        color += reflectance * reflection_color


                # Apply indirect lighting (color bleeding)
                hemisphere_dir = sample_hemisphere(intersection.normal)
                indirect_ray = hc.Ray(intersection.position + 1e-5 * intersection.normal, hemisphere_dir)
                indirect_color = path_trace(indirect_ray, depth + 1, max_depth)

                # Scale by material's diffuse and cosine factor
                # diff = max(glm.dot(intersection.normal, light_dir), 0.0)
                cosine_factor = max(glm.dot(intersection.normal, hemisphere_dir), 0.0)
                color += intersection.mat.diffuse * indirect_color * cosine_factor / math.pi
                
                # Refraction (path trace the refracted ray if needed)
                if intersection.mat.refraction_index > 1.0:
                    eta_ratio = (1.0 / intersection.mat.refraction_index 
                                if glm.dot(ray.direction, intersection.normal) < 0
                                else intersection.mat.refraction_index / 1.0)

                    normal = intersection.normal if glm.dot(ray.direction, intersection.normal) < 0 else -intersection.normal

                    # Compute refracted direction using Snell's Law
                    refracted_dir = refract(ray.direction, normal, eta_ratio)
                    if refracted_dir is not None:
                        refracted_ray = hc.Ray(intersection.position - 1e-5 * normal, refracted_dir)
                        refraction_color = path_trace(refracted_ray, depth + 1, max_depth)
                        color += (1.0 - reflectance) * refraction_color

            return glm.clamp(color, glm.vec3(0.0), glm.vec3(1.0))  # Ensure the color is clamped to [0, 1]

        
        # Function to sample the hemisphere for indirect lighting (color bleeding)
        def sample_hemisphere(normal):
            # Generate two orthogonal vectors to the normal
            if abs(normal.x) > abs(normal.z):
                tangent = glm.vec3(-normal.y, normal.x, 0.0)
            else:
                tangent = glm.vec3(0.0, -normal.z, normal.y)
            tangent = glm.normalize(tangent)
            bitangent = glm.cross(normal, tangent)

            # Uniformly sample the hemisphere
            u = random.uniform(0.0, 1.0)
            v = random.uniform(0.0, 1.0)
            theta = math.acos(math.sqrt(u))  # Cosine-weighted
            phi = 2.0 * math.pi * v

            x = math.sin(theta) * math.cos(phi)
            y = math.sin(theta) * math.sin(phi)
            z = math.cos(theta)

            # Transform the local direction to world space
            local_dir = glm.vec3(x, y, z)
            world_dir = glm.mat3(tangent, bitangent, normal) * local_dir
            return glm.normalize(world_dir)


        # Function to sample the area light for soft shadows
        def sample_area_light(light, i, j, num_samples):
            """
            Stratified sampling on the area light surface.
            Splits the light into a grid and samples within each cell.
            """
            u = (i + glm.linearRand(0.0, 1.0)) / num_samples
            v = (j + glm.linearRand(0.0, 1.0)) / num_samples

            # Map (u, v) to the light's physical dimensions
            half_width = light.size.x * 0.5
            half_height = light.size.y * 0.5

            sample_x = light.position.x - half_width + u * light.size.x
            sample_y = light.position.y
            sample_z = light.position.z - half_height + v * light.size.y

            return glm.vec3(sample_x, sample_y, sample_z)
        
         # TODO: Perform shading computations on the intersection point
         # Compute the shadings (including shadows) at the intersection point
         # Returns the color of the pixel at the intersection point which is the sum of the ambient, diffuse and specular components
        def compute_shading(intersection, ray, depth, max_depth):
            # Ambient shading (common for both ray tracing and path tracing)
            color = self.ambient * intersection.mat.diffuse

            # Loop through each light in the scene (same for both ray tracing and path tracing)
            for light in self.lights:

                # The following section is for area lights
                if isinstance(light, hc.AreaLight):
                    num_samples = 64 #36  # Quality vs. performance tradeoff. I usually set it to 16 or 25 or 36 for more complex renders or else my code takes days to render
                    strata = int(glm.sqrt(num_samples))  # Determine strata dimension based on samples

                    light_contribution = glm.vec3(0.0)
                    for sample in range(num_samples):
                    # for sample_point in light_samples:
                        sample_point = sample_area_light(light, sample % strata, sample // strata, strata)
                        #or sample_point = light.sample()
                        light_dir = glm.normalize(sample_point - intersection.position)
                        distance_to_light = glm.length(sample_point - intersection.position)

                        # Shadow ray check
                        shadow_ray = hc.Ray(intersection.position + 1e-5 * intersection.normal, light_dir)
                        shadow_intersection = test_intersections(shadow_ray)
                        if shadow_intersection and shadow_intersection.t < distance_to_light:
                            continue

                        # Lambertian diffuse shading
                        diff = max(glm.dot(intersection.normal, light_dir), 0.0)
                        diffuse = diff * light.colour * intersection.mat.diffuse

                        # Blinn-Phong specular shading
                        view_dir = glm.normalize(self.eye_position - intersection.position)
                        half_vector = glm.normalize(light_dir + view_dir)
                        spec_intensity = max(glm.dot(intersection.normal, half_vector), 0.0)
                        specular = (
                            (spec_intensity ** intersection.mat.shininess)
                            * intersection.mat.specular
                            * light.colour
                        ) if glm.length(intersection.mat.specular) > 1e-6 else glm.vec3(0.0)

                        light_contribution += diffuse + specular

                    # Normalize by the number of samples and scale by the light's area
                    light_area = light.size.x * light.size.y
                    light_contribution /= (num_samples*light_area)
                    color += light_contribution

                # For point lights (so no area lights and no path tracing)
                else: 
                    light_dir = glm.normalize(light.vector - intersection.position)
                    distance_to_light = glm.length(light.vector - intersection.position)

                    # Attenuation for point light
                    attenuation = 1.0 / (light.attenuation[0] * (distance_to_light ** 2) +
                                        light.attenuation[1] * distance_to_light +
                                        light.attenuation[2])
                    light_intensity = light.colour * attenuation

                    # Shadow ray to check if the light is blocked
                    shadow_ray = hc.Ray(intersection.position + 1e-4 * intersection.normal, light_dir)
                    shadow_intersection = test_intersections(shadow_ray)
                    if shadow_intersection and shadow_intersection.t < distance_to_light:
                        continue

                    # Diffuse lighting (Lambertian reflection)
                    diff = max(glm.dot(intersection.normal, light_dir), 0.0)
                    diffuse = diff * light_intensity * intersection.mat.diffuse
                    color += diffuse

                    # Specular lighting (Blinn-Phong reflection)
                    view_dir = glm.normalize(self.eye_position - intersection.position)
                    half_vector = glm.normalize(light_dir + view_dir)
                    spec_intensity = max(glm.dot(intersection.normal, half_vector), 0.0)
                    if glm.length(intersection.mat.specular) > 1e-6:
                        specular = (spec_intensity ** intersection.mat.shininess) * intersection.mat.specular * light_intensity
                        color += specular

            return glm.clamp(color, glm.vec3(0.0), glm.vec3(1.0))


        # Perform ray tracing/path tracing based on the flag
        # Loop through each pixel in the image
        # This is where we call the functions to generate rays, test for intersections and compute shading
        # Perform ray/path tracing with Anti-Aliasing and Super-Sampling
        for col in tqdm(range(self.width)):
            for row in range(self.height):
                color = glm.vec3(0, 0, 0)  # Initialize color to black

                subgrid_size = self.samples  # AA_samples is the subgrid dimension
                total_samples = subgrid_size * subgrid_size  # Total samples per pixel
                # For super-sampling and anti-aliasing, we sample multiple rays per pixel
                for sub_y in range(subgrid_size):
                    for sub_x in range(subgrid_size):
                        # Calculate the offsets for the subgrid within the pixel
                        jitter_x = jitter_y = 0.0
                        if self.jitter:
                            jitter_x = glm.linearRand(-0.5, 0.5) #0.25
                            jitter_y = glm.linearRand(-0.5, 0.5) #0.25

                        offset_x = (sub_x + 0.5 + jitter_x) / subgrid_size
                        offset_y = (sub_y + 0.5 + jitter_y) / subgrid_size

                        ray = generate_ray(col, row, offset_x, offset_y)
                        if self.path_tracing:
                            color += path_trace(ray)  # Use path tracing if enabled
                        else:
                            color += ray_trace(ray)  # Use ray tracing (you should already have this)
                color /= total_samples

                # Clamp the color values to be within [0, 1]
                image[row, col, 0] = max(0.0, min(1.0, color.x))
                image[row, col, 1] = max(0.0, min(1.0, color.y))
                image[row, col, 2] = max(0.0, min(1.0, color.z))
        return image

        



