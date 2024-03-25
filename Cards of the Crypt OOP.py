import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import random
import glm
from PIL import Image

WIN_WIDTH  = 800
WIN_HEIGHT = 600

class Camera:
    def __init__(self, position=glm.vec3(0, 0, 3), yaw=0.0, pitch=0.0, speed=0.1):
        self.position = position
        self.yaw = yaw
        self.pitch = pitch
        self.speed = speed
        self.rotation_speed = 1.0
        self.front = glm.vec3(0, 0, -1)
        self.update_vectors()

    def get_view_matrix(self):
        return glm.lookAt(self.position, self.position + self.front, glm.vec3(0, 1, 0))

    def update_vectors(self):
        x = glm.cos(glm.radians(self.yaw)) * glm.cos(glm.radians(self.pitch))
        y = glm.sin(glm.radians(self.pitch))
        z = glm.sin(glm.radians(self.yaw)) * glm.cos(glm.radians(self.pitch))

        self.front = glm.normalize(glm.vec3(x, y, z))
        self.right = glm.normalize(glm.cross(self.front, glm.vec3(0, 1, 0)))
        self.up = glm.normalize(glm.cross(self.right, self.front))

    def move_forward(self):
        self.position += self.speed * self.front

    def move_backward(self):
        self.position -= self.speed * self.front

    def strafe_left(self):
        self.position -= self.speed * self.right

    def strafe_right(self):
        self.position += self.speed * self.right
        
    def rotate_left(self):
        self.yaw -= self.rotation_speed
        self.update_vectors()

    def rotate_right(self):
        self.yaw += self.rotation_speed
        self.update_vectors()

class Shader:
    def __init__(self, vertex_shader_source, fragment_shader_source):
        self.program = self.compile_and_link_program(vertex_shader_source, fragment_shader_source)

    def read_file(filename):
        with open(filename, 'r') as f:
            return f.read()

    def compile_shader(self, source, shader_type):
        try:
            shader = glCreateShader(shader_type)
            glShaderSource(shader, source)
            glCompileShader(shader)
            success = glGetShaderiv(shader, GL_COMPILE_STATUS)

            if not success:
                error_log = glGetShaderInfoLog(shader).decode('utf-8') 
                raise RuntimeError(f"Shader compilation failed:\n{error_log}") 
            else:
                print(f"Shader compiled successfully (type: {shader_type})")

        except RuntimeError as error:
            print(f"Shader Error: {error}")
            return None

        return shader 

    def compile_and_link_program(self, vertex_source, fragment_source):
        vertex_shader = self.compile_shader(vertex_source, GL_VERTEX_SHADER)
        fragment_shader = self.compile_shader(fragment_source, GL_FRAGMENT_SHADER)
        shader_program = glCreateProgram()

        glAttachShader(shader_program, vertex_shader)
        glAttachShader(shader_program, fragment_shader)
        glLinkProgram(shader_program)

        success = glGetProgramiv(shader_program, GL_LINK_STATUS)
        if not success:
            error_log = glGetProgramInfoLog(shader_program).decode('utf-8')
            raise RuntimeError(f"Error linking shader program:\n{error_log}")     
        else:
            print("Shader program linked successfully")

        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)

        return shader_program

    def use(self):
        glUseProgram(self.program)

class Texture:
    def __init__(self, image_path):
        self.texture_id = self.load_texture(image_path)

    def load_texture(self, image_path):
        try:
            image = Image.open(image_path)
            image_data = np.array(list(image.getdata()), np.uint8)

            texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture)
            print("Texture ID:", texture)

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT) 
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR) 
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR) 

            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data) 

            return texture 
        
        except (FileNotFoundError, AttributeError):
            print(f"Error loading texture: {image_path}")
            return None

    def bind(self):
        glBindTexture(GL_TEXTURE_2D, self.texture_id)

class Tile:
    def __init__(self, texture, is_wall=False):
        self.texture = texture
        self.is_wall = is_wall

class DungeonRoom:
    def __init__(self, width, height, wall_texture, floor_texture):
        self.width = width
        self.height = height
        self.tiles = [[Tile(None) for _ in range(width)] for _ in range(height)] 
        self.wall_texture = wall_texture
        self.floor_texture = floor_texture 
        self.generate_room() 
        self.connect_rooms()

    def generate_room(self):
        min_room_size = 4 
        max_room_size = 8  

        for y in range(self.height):
            for x in range(self.width):
                if x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1:
                    self.tiles[y][x].texture = self.wall_texture
                    self.tiles[y][x].is_wall = True
                else:
                    if random.randint(0, 3) == 0: 
                        room_width = random.randint(min_room_size, max_room_size)
                        room_height = random.randint(min_room_size, max_room_size)
                        room_x_offset = random.randint(1, max(self.width - room_width - 2, 1)) 
                        room_y_offset = random.randint(1, max(self.height - room_height - 2, 1)) 

                        for ry in range(room_y_offset, room_y_offset + room_height):
                            for rx in range(room_x_offset, room_x_offset + room_width):
                                self.tiles[ry][rx].texture = self.floor_texture 

    def find_rooms(self):
        rooms = []
        for y in range(self.height):
            for x in range(self.width):
                if self.tiles[y][x].texture == self.floor_texture:  # Find a floor tile
                    center_x = x + (self.find_room_width(x, y) // 2)  # Approx. center
                    center_y = y + (self.find_room_height(x, y) // 2)
                    rooms.append((center_x, center_y))
        return rooms

    def find_room_width(self, start_x, start_y):  # Helper for find_rooms
        for x in range(start_x, self.width):
            if self.tiles[start_y][x].texture != self.floor_texture:
                return x - start_x
        return self.width - start_x

    def find_room_height(self, start_x, start_y):  # Helper for find_rooms
        for y in range(start_y, self.height):
            if self.tiles[y][start_x].texture != self.floor_texture:
                return y - start_y
        return self.height - start_y

    def connect_rooms(self):
        rooms = self.find_rooms() 

        for i in range(len(rooms) - 1):
            start = rooms[i]
            end = rooms[i + 1]

            for x in range(start[0], end[0]):
                self.tiles[start[1]][x].texture = self.floor_texture

            for y in range(start[1], end[1]):
                self.tiles[y][end[0]].texture = self.floor_texture

def init_game():
    pygame.init()
    display = (WIN_WIDTH, WIN_HEIGHT)

    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    glEnable(GL_DEPTH_TEST) 
    glDepthFunc(GL_LEQUAL)
    print(glGetString(GL_VERSION))

    #VERTICES
    vertices = np.array([
    # Front face 
    -0.5, -0.5,  0.5, 0.0, 0.0, # Bottom-left
     0.5, -0.5,  0.5, 1.0, 0.0, # Bottom-right 
     0.5,  0.5,  0.5, 1.0, 1.0, # Top-right
    -0.5,  0.5,  0.5, 0.0, 1.0, # Top-left

    # Back face
    -0.5, -0.5, -0.5, 0.0, 0.0,  
     0.5, -0.5, -0.5, 1.0, 0.0,  
     0.5,  0.5, -0.5, 1.0, 1.0,  
    -0.5,  0.5, -0.5, 0.0, 1.0,  

    # Left face
    -0.5, -0.5, -0.5, 0.0, 0.0, 
    -0.5,  0.5, -0.5, 1.0, 0.0, 
    -0.5,  0.5,  0.5, 1.0, 1.0, 
    -0.5, -0.5,  0.5, 0.0, 1.0,

    # Right face
     0.5, -0.5,  0.5, 0.0, 0.0, 
     0.5,  0.5,  0.5, 1.0, 0.0, 
     0.5,  0.5, -0.5, 1.0, 1.0, 
     0.5, -0.5, -0.5, 0.0, 1.0,

    # Top face
    -0.5,  0.5,  0.5, 0.0, 0.0, 
     0.5,  0.5,  0.5, 1.0, 0.0,
     0.5,  0.5, -0.5, 1.0, 1.0,
    -0.5,  0.5, -0.5, 0.0, 1.0,

    # Bottom face
    -0.5, -0.5, -0.5, 0.0, 0.0, 
     0.5, -0.5, -0.5, 1.0, 0.0,
     0.5, -0.5,  0.5, 1.0, 1.0,
    -0.5, -0.5,  0.5, 0.0, 1.0,
    ], dtype=np.float32)

    # VBO setup
    print("Configuring VBO...")
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo) 
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)  # Position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * vertices.itemsize, None)  
    glEnableVertexAttribArray(1)  # Texture coords
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * vertices.itemsize, ctypes.c_void_p(3 * vertices.itemsize)) 
    print("VBO configured!")

    #SHADERS
    print("Creating shader...")
    vertex_shader_source = Shader.read_file("vertex_shader.vs")
    fragment_shader_source = Shader.read_file("fragment_shader.frag")
    shader = Shader(vertex_shader_source, fragment_shader_source)
    print("Shader created:", shader) 

    #TEXTURES
    print("Loading textures...")
    wall_texture = Texture("wall_texture.jpg")
    if wall_texture is None:
        print("No Wall Texture!'")

    floor_texture = Texture("floor_texture.jpeg")
    if floor_texture is None:
        print("No Floor Texture!")

    print("Textures loaded.") 

    #CAMERA
    print("Creating camera...")
    camera = Camera(position=glm.vec3(0, 0, 5), speed=0.05)
    print("Camera created:", camera)
    print("Camera Position:", camera.position)

    #ROOM
    print("Generating room...") 
    room = DungeonRoom(10, 10, wall_texture, floor_texture)
    print("Room generated.") 

    return shader, camera, room, wall_texture, floor_texture

def handle_input(camera):
    keys = pygame.key.get_pressed()

    if keys[K_w]:
        camera.move_forward()
    if keys[K_s]:
        camera.move_backward()
    if keys[K_a]:
        camera.strafe_left()
    if keys[K_d]:
        camera.strafe_right()
    if keys[K_q]:
        camera.rotate_left()
    if keys[K_e]:
        camera.rotate_right()

def update(camera):
    handle_input(camera)

def draw(shader, camera, room, wall_texture, floor_texture):
    tile_size = 1.0 
    
    glClearColor(0.2, 0.3, 0.3, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    shader.use()

    model_matrix = glm.mat4(1.0)  
    view_matrix = camera.get_view_matrix()
    projection_matrix = glm.perspective(glm.radians(60.0), (WIN_WIDTH / WIN_HEIGHT), 0.1, 100.0)

    glUniformMatrix4fv(glGetUniformLocation(shader.program, "model"), 1, GL_FALSE, glm.value_ptr(model_matrix))
    glUniformMatrix4fv(glGetUniformLocation(shader.program, "view"), 1, GL_FALSE, glm.value_ptr(view_matrix))
    glUniformMatrix4fv(glGetUniformLocation(shader.program, "projection"), 1, GL_FALSE, glm.value_ptr(projection_matrix))
    glUniform1f(glGetUniformLocation(shader.program, "tileSize"), tile_size)

    for y in range(room.height):
        for x in range(room.width):
            tile = room.tiles[y][x]

            if tile.texture == wall_texture:
                wall_texture.bind()
            elif tile.texture == floor_texture:
                floor_texture.bind()

            tile_x = x
            tile_y = 0
            tile_z = -y

            tile_model_matrix = glm.translate(glm.mat4(1.0), glm.vec3(tile_x, tile_y, tile_z)) 

            glUniformMatrix4fv(glGetUniformLocation(shader.program, "model"), 1, GL_FALSE, glm.value_ptr(tile_model_matrix))

            glDrawArrays(GL_QUADS, 0, 4) 

# For Debugging
'''def draw(shader, camera, room, wall_texture):
    glClearColor(0.2, 0.3, 0.3, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    shader.use()

    model_matrix = glm.mat4(1.0)  # Identity matrix (no transformations)
    view_matrix = camera.get_view_matrix()
    projection_matrix = glm.perspective(glm.radians(60.0), (WIN_WIDTH / WIN_HEIGHT), 0.1, 100.0)

    glUniformMatrix4fv(glGetUniformLocation(shader.program, "model"), 1, GL_FALSE, glm.value_ptr(model_matrix))
    glUniformMatrix4fv(glGetUniformLocation(shader.program, "view"), 1, GL_FALSE, glm.value_ptr(view_matrix))
    glUniformMatrix4fv(glGetUniformLocation(shader.program, "projection"), 1, GL_FALSE, glm.value_ptr(projection_matrix))

    glActiveTexture(GL_TEXTURE0)
    wall_texture.bind()

    glDrawArrays(GL_QUADS, 0, 4)'''


if __name__ == "__main__":
    shader, camera, room, wall_texture, floor_texture= init_game()
    
    while True:  
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        update(camera) 
        draw(shader, camera, room, wall_texture, floor_texture) 
        pygame.display.flip()  
        pygame.time.wait(10) 