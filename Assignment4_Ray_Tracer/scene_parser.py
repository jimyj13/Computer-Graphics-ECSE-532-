# Yong-Jung Song 260735417

import json
import helperclasses as hc
import geometry as geom
import scene
import glm
import numpy as np

def make_vec3(array: list):
    return glm.vec3(array[0], array[1], array[2])

def make_matrix( t: glm.vec3, r: glm.vec3, s: glm.vec3 ):
    M = glm.mat4(1.0)
    M = glm.translate( M, t )
    M = glm.rotate( M, glm.radians(r.x), glm.vec3(1, 0, 0) )
    M = glm.rotate( M, glm.radians(r.y), glm.vec3(0, 1, 0) )
    M = glm.rotate( M, glm.radians(r.z), glm.vec3(0, 0, 1) )
    M = glm.scale( M, s )     
    return M   

def load_scene(infile: str):
    print("Parsing file:", infile)
    f = open(infile)
    data = json.load(f)

    # Loading camera
    cam_pos = make_vec3(data["camera"]["position"])
    cam_lookat = make_vec3(data["camera"]["lookAt"])
    cam_up = make_vec3(data["camera"]["up"])
    cam_fov = data["camera"]["fov"]

    # Code for the DOF blur implementation   
    # Loading DOF parameters (default values if not found)
    aperture_radius = data["camera"].get("aperture_radius", 0.0)  # Load aperture radius from camera section
    focal_distance = data["camera"].get("focal_distance", 1.0)    # Load focal distance from camera section

    # Loading resolution
    default_resolution = [1280, 720]    
    width = data.get("resolution", default_resolution)[0]
    height = data.get("resolution", default_resolution)[1]
        
    # Loading ambient light
    ambient = make_vec3(data.get("ambient", [0.1, 0.1, 0.1])) # set a reasonable default ambient light

    # Loading Anti-Aliasing options    
    jitter = data.get( "AA_jitter", False ) # default to no jitter
    samples = data.get( "AA_samples", 1 ) # default to no supersampling

    # Code for the path tracing implementation
    # Loading path tracing settings (backward compatibility)
    path_tracing = data.get("path_tracing", False)

    # Code for the area lights implementation
    # Loading area lights (soft shadows)
    area_lights = data.get("area_lights", False)
    
    
    # Loading scene lights
    lights = []    
    for light in data.get("lights", []):
        l_type = light["type"]
        l_name = light["name"]
        l_colour = make_vec3(light["colour"])            
        l_power = light.get( "power", 1.0 ) # The power scales the specified light colour

        if l_type == "point":
            # For point lights, the position is required
            l_vector = make_vec3(light["position"])
            l_attenuation = glm.vec3(1, 0, 0) if "attenuation" not in light else make_vec3(light["attenuation"])
            lights.append(hc.Light(l_type, l_name, l_colour * l_power, l_vector, l_attenuation))

        elif l_type == "directional":
            # For directional lights, use the direction instead of position
            l_vector = glm.normalize(make_vec3(light["direction"]))
            l_attenuation = glm.vec3(0, 0, 0)
            if "attenuation" in light:
                print("Directional light", l_name, "has attenuation, ignoring")
            lights.append(hc.Light(l_type, l_name, l_colour * l_power, l_vector, l_attenuation))

        # Code for the area lights implementation
        elif l_type == "area":
            # For area lights, ensure both position and size are present (defaults for missing size)
            l_position = make_vec3(light["position"])  # Position of the area light
            l_size = make_vec3(light.get("size", [1.0, 1.0, 1.0]))  # Default to 1x1x1 if size is missing
            lights.append(hc.AreaLight(l_type, l_name, l_colour * l_power, l_position, l_size))

        else:
            print("Unknown light type", l_type, ", skipping initialization")
            continue

    # Loading materials
    material_by_name = {} # materials dictionary 
    for material in data["materials"]:
        mat_name = material["name"]
        mat_diffuse = make_vec3(material["diffuse"])
        mat_specular = make_vec3(material["specular"])
        mat_shininess = 0 if "shininess" not in material else material["shininess"]

        # Added reflection and refraction indexes
        mat_reflection_index = material.get("reflection_index", 1.0)  # Default to 1.0 (no reflection)
        mat_refraction_index = material.get("refraction_index", 1.0)  # Default to 1.0 (no refraction)
    
        # Create the Material object
        material_by_name[mat_name] = hc.Material(mat_name, mat_diffuse, mat_specular, mat_shininess, mat_reflection_index, mat_refraction_index)

    # load geometries
    objects = [] # list of loaded object geometries and hierarchy roots
    geometry_by_name = {} # dictionary of geometries by name (for instances)

    for geometry in data["objects"]:
        g = load_geometry( geometry, material_by_name, geometry_by_name )
        objects.append(g)
        geometry_by_name[g.name] = g

    return scene.Scene(width, height, jitter, samples,  # General settings
                cam_pos, cam_lookat, cam_up, cam_fov,  # Camera settings
                ambient, lights,  # Light settings
                # DOF parameters
                objects, aperture_radius, focal_distance,
                # Path tracing and area lights parameters
                path_tracing=path_tracing, area_lights=area_lights)

def load_geometry( geometry, material_by_name, geometry_by_name ):

    # Elements common to all objects: name, type, and material(s)
    g_name = geometry["name"]
    g_type = geometry["type"]
    g_mats = [ material_by_name[mat] for mat in geometry.get("materials",[]) ]

    if g_type == "sphere":
        g_pos = make_vec3(geometry.get("position", [0, 0, 0]))
        g_radius = geometry["radius"]
        return geom.Sphere(g_name, g_type, g_mats, g_pos, g_radius)
    elif g_type == "plane":
        g_pos = make_vec3(geometry.get("position", [0, 0, 0]))
        g_normal = make_vec3(geometry["normal"])
        return geom.Plane(g_name, g_type, g_mats, g_pos, g_normal)
    elif g_type == "box":
        minpos = make_vec3(geometry.get("min",[-1,-1,-1]))
        maxpos = make_vec3(geometry.get("max",[1,1,1]))
        return geom.AABB(g_name, g_type, g_mats, minpos, maxpos)
    elif g_type == "mesh":
        g_path = geometry["filepath"]
        g_pos = make_vec3(geometry.get("position", [0, 0, 0]))
        g_scale = geometry["scale"]
        return geom.Mesh(g_name, g_type, g_mats, g_pos, g_scale, g_path)
    elif g_type == "instance":
        g_pos = make_vec3(geometry.get("position", [0, 0, 0]))
        g_r = make_vec3(geometry.get("rotation", [0, 0, 0]))
        g_s = make_vec3(geometry.get("scale", [1, 1, 1]))
        M = make_matrix(g_pos, g_r, g_s)
        node = geom.Node(g_name, g_type, M, g_mats)
        node.children.append( geometry_by_name[geometry["ref"]] )
        return node
    elif g_type == "node":
        g_pos = make_vec3(geometry.get("position", [0, 0, 0]))
        g_r = make_vec3(geometry.get("rotation", [0, 0, 0]))
        g_s = make_vec3(geometry.get("scale", [1, 1, 1]))
        M = make_matrix(g_pos, g_r, g_s)
        node = geom.Node(g_name, g_type, M, g_mats)
        for child in geometry["children"]:
            g = load_geometry(child, material_by_name, geometry_by_name)
            node.children.append(g)
            geometry_by_name[g.name] = g
        return node

    # Code for rendering large meshes using acceleration with BVH
    elif g_type == "large_mesh":
        g_path = geometry["filepath"] 
        g_pos = make_vec3(geometry.get("position", [0, 0, 0])) 
        g_scale = geometry["scale"]
        return geom.LargeMeshes(g_name, g_type, g_mats, g_pos, g_scale, g_path)

    else:
        print("Unkown object type", g_type, ", skipping initialization")
        return None