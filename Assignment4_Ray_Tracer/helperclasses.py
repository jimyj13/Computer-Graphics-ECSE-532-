# Yong-Jung Song 260735417

import glm
import random

class Ray:
    def __init__(self, o: glm.vec3, d: glm.vec3):
        self.origin = o
        self.direction = d

    def getDistance(self, point: glm.vec3):
        return glm.length(point - self.origin)

    def getPoint(self, t: float):
        return self.origin + self.direction * t     

class Material:
    def __init__(self, name: str, diffuse: glm.vec3, specular: glm.vec3, shininess: float, reflection_index: float = 1.0, refraction_index: float = 1.0):
        self.name = name
        self.diffuse = diffuse      # kd diffuse coefficient
        self.specular = specular    # ks specular coefficient
        self.shininess = shininess  # specular exponent
        self.reflection_index = reflection_index  # refractive index for refraction and reflection
        self.refraction_index = refraction_index  # refractive index for refraction and reflection

class Light:
    def __init__(self, ltype: str, name: str, colour: glm.vec3, vector: glm.vec3, attenuation: glm.vec3):
        self.name = name
        self.type = ltype       # type is either "point" or "directional"
        self.colour = colour    # colour and intensity of the light
        self.vector = vector    # position, or normalized direction towards light, depending on the light type
        self.attenuation = attenuation # attenuation coeffs [quadratic, linear, constant] for point lights

# class defined for Area Lights implementation
class AreaLight(Light):
    def __init__(self, ltype: str, name: str, colour: glm.vec3, position: glm.vec3, size: glm.vec3):
        super().__init__(ltype, name, colour, position, glm.vec3(0, 0, 0))  # No attenuation for area lights
        self.size = size  # The size of the area light (width, height)
        self.position = position  # The position of the area light

        # Create local basis vectors for the rectangle (u and v are perpendicular)
        self.u_vec = glm.vec3(1.0, 0.0, 0.0) * size.x  # Default to x-axis aligned
        self.v_vec = glm.vec3(0.0, 1.0, 0.0) * size.y  # Default to y-axis aligned

    # Stratified sampling for area lights, used in the scene.py    
    def sample(self, strata: int):
        """Stratified sampling to get all points on the area light."""
        samples = []
        strata_size = 1.0 / strata  # Divide the light surface into smaller strata

        for i in range(strata):
            for j in range(strata):
                u = (i + glm.linearRand(0.0, 1.0)) * strata_size
                v = (j + glm.linearRand(0.0, 1.0)) * strata_size
                # Map to the area light rectangle
                sample_point = self.position + (u - 0.5) * self.u_vec + (v - 0.5) * self.v_vec
                samples.append(sample_point)
        return samples



class Intersection:
    def __init__(self, t: float, normal: glm.vec3, position: glm.vec3, material: Material):
        self.t = t
        self.normal = normal
        self.position = position
        self.mat = material

    @staticmethod
    def default(): # create an empty intersection record with t = inf
        t = float("inf")
        normal = None 
        position = None 
        mat = None 
        return Intersection(t, normal, position, mat)
