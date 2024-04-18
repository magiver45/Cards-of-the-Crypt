import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import glm

WIN_WIDTH  = 800
WIN_HEIGHT = 600

class Camera:
    def __init__(self, position=glm.vec3(0, 0, 0), yaw=0.0, pitch=0.0, speed=0.1, rotation_speed=1.0):
        self.position = position
        self.yaw = yaw
        self.pitch = pitch
        self.speed = speed
        self.rotation_speed = rotation_speed
        self.front = glm.vec3(0, 0, 0)
        self.update_vectors()

    def get_view_matrix(self):
        return glm.lookAt(self.position, self.position + self.front, glm.vec3(0, 1, 0))

    def update_vectors(self):
        x = glm.cos(glm.radians(self.yaw)) * glm.cos(glm.radians(self.pitch))
        y = glm.sin(glm.radians(self.pitch))
        z = glm.sin(glm.radians(self.yaw)) * glm.cos(glm.radians(self.pitch))

        self.front = glm.normalize(glm.vec3(x, y, z))
        self.right = glm.normalize(glm.cross(self.front, glm.vec3(0, 1, 0)))
        self.up    = glm.normalize(glm.cross(self.right, self.front))

    def move_forward(self):
        self.position += self.speed * self.front

    def sprint(self):
        self.position += self.speed * self.front * 2

    def move_backward(self):
        self.position -= self.speed/2 * self.front

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
    loaded_textures = {}

    def __init__(self, image_path):
        if image_path in Texture.loaded_textures:
            self.texture = Texture.loaded_textures[image_path]
        else:
            try:
                image = pygame.image.load(image_path).convert_alpha()
                image_width, image_height = image.get_rect().size
                image_data = pygame.image.tostring(image, "RGBA")

                self.texture = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, self.texture)
                print("Texture ID:", self.texture)

                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT) 
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR) 
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR) 

                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)

                Texture.loaded_textures[image_path] = self.texture  # Cache the loaded texture
            
            except (FileNotFoundError, AttributeError):
                print(f"Error loading texture: {image_path}")
                return None

    def bind(self):
        glBindTexture(GL_TEXTURE_2D, self.texture)

class Tile:
    def __init__(self, texture, is_wall=False):
        self.texture = texture
        self.is_wall = is_wall

class DungeonRoom:
    def __init__(self, width, height, depth, wall_texture, floor_texture):
        self.width = width
        self.height = height
        self.depth = depth
        self.tiles = [[Tile(None) for _ in range(width)] for _ in range(height)] 
        self.wall_texture = wall_texture
        self.floor_texture = floor_texture 
        self.generate_room()

    def generate_room(self):  
        for y in range(self.height):
            for x in range(self.width):
                if x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1:
                    self.tiles[y][x].texture = self.wall_texture
                    self.tiles[y][x].is_wall = True
                else:
                    self.tiles[y][x].texture = self.floor_texture
                    self.tiles[y][x].is_wall = False

class SetupGame:
    def __init__(self):
        self.init_pygame()
        self.vertices = self.create_vertices()
        self.init_vbo(self.vertices)
        self.shader = self.init_shader()
        self.wall_texture, self.floor_texture = self.init_textures()
        self.camera = self.init_camera()
        self.room = self.init_room(self.wall_texture, self.floor_texture)

    def init_pygame(self):
        """Initialize Pygame and OpenGL."""
        pygame.init()
        display = (WIN_WIDTH, WIN_HEIGHT)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        print(glGetString(GL_VERSION))

    def create_vertices(self):
        """Create vertices for a unit cube."""
        vertices = np.array([
            # Front face
            -0.5, -0.5,  0.5, 1.0, 0.0, # left  bottom front
             0.5, -0.5,  0.5, 0.0, 0.0, # right bottom front
             0.5,  0.5,  0.5, 0.0, 1.0, # right top    front
            -0.5,  0.5,  0.5, 1.0, 1.0, # left  top    front

            # Back face
            -0.5, -0.5, -0.5, 0.0, 0.0, # left  bottom back
             0.5, -0.5, -0.5, 1.0, 0.0, # right bottom back
             0.5,  0.5, -0.5, 1.0, 1.0, # right top    back
            -0.5,  0.5, -0.5, 0.0, 1.0, # left  top    back

            # Left face
            -0.5, -0.5, -0.5, 1.0, 0.0, # left bottom back
            -0.5,  0.5, -0.5, 1.0, 1.0, # left top    back
            -0.5,  0.5,  0.5, 0.0, 1.0, # left top    front
            -0.5, -0.5,  0.5, 0.0, 0.0, # left bottom front

            # Right face
             0.5, -0.5,  0.5, 1.0, 0.0, # right bottom front
             0.5,  0.5,  0.5, 1.0, 1.0, # right top    front
             0.5,  0.5, -0.5, 0.0, 1.0, # right top    back
             0.5, -0.5, -0.5, 0.0, 0.0, # right bottom back

            # Top face
            -0.5,  0.5,  0.5, 0.0, 0.0, # left  top    front
             0.5,  0.5,  0.5, 1.0, 0.0, # right top    front
             0.5,  0.5, -0.5, 1.0, 1.0, # right top    back
            -0.5,  0.5, -0.5, 0.0, 1.0, # left  top    back

            # Bottom face
            -0.5, -0.5, -0.5, 0.0, 0.0, # left  bottom back
             0.5, -0.5, -0.5, 1.0, 0.0, # right bottom back
             0.5, -0.5,  0.5, 1.0, 1.0, # right bottom front
            -0.5, -0.5,  0.5, 0.0, 1.0, # left  bottom front
        ], dtype=np.float32)

        return vertices

    def init_vbo(self, vertices):
        """Initialize Vertex Buffer Object (VBO)."""
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)  # Position
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * vertices.itemsize, None)
        glEnableVertexAttribArray(1)  # Texture coords
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * vertices.itemsize, ctypes.c_void_p(3 * vertices.itemsize))

    def init_shader(self):
        """Initialize shader program."""
        vertex_shader_source = Shader.read_file("vertex_shader.vs")
        fragment_shader_source = Shader.read_file("fragment_shader.frag")
        return Shader(vertex_shader_source, fragment_shader_source)

    def init_textures(self):
        """Load and initialize textures."""
        wall_texture = Texture("wall_texture.jpg")
        floor_texture = Texture("floor_texture.jpeg")
        if wall_texture is None or floor_texture is None:
            raise RuntimeError("Failed to load textures")
        return wall_texture, floor_texture

    def init_camera(self):
        """Initialize the camera."""
        return Camera(position=glm.vec3(5, 0, -5), rotation_speed=2.5)

    def init_room(self, wall_texture, floor_texture):
        """Initialize the dungeon room."""
        return DungeonRoom(10, 10, 3, wall_texture, floor_texture)

def handle_input(camera):
    keys = pygame.key.get_pressed()

    if keys[K_w]:
        camera.move_forward()
        if keys[K_LSHIFT]:
            camera.sprint()
    if keys[K_s]:
        camera.move_backward()
    if keys[K_q]:
        camera.strafe_left()
    if keys[K_e]:
        camera.strafe_right()
    if keys[K_a]:
        camera.rotate_left()
    if keys[K_d]:
        camera.rotate_right()



def draw(shader, camera, vertices, room, wall_texture, floor_texture):
    tile_size = 1.0
    
    glClearColor(0.2, 0.3, 0.3, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    shader.use()
  
    view_matrix = camera.get_view_matrix()
    projection_matrix = glm.perspective(glm.radians(60.0), (WIN_WIDTH / WIN_HEIGHT), 0.1, 100.0)

    glUniformMatrix4fv(glGetUniformLocation(shader.program, "view"), 1, GL_FALSE, glm.value_ptr(view_matrix))
    glUniformMatrix4fv(glGetUniformLocation(shader.program, "projection"), 1, GL_FALSE, glm.value_ptr(projection_matrix))
    glUniform1f(glGetUniformLocation(shader.program, "tileSize"), tile_size)

    # Batch tiles with the same texture
    for texture, tiles in ((wall_texture, []), (floor_texture, [])):
        for z in range(room.depth):
            for y in range(room.height):
                for x in range(room.width):
                    tile = room.tiles[y][x]

                    if tile.texture == texture:
                        tiles.append((x, y, z))

        if tiles:
            texture.bind()
            for tile in tiles:
                x, y, z = tile
                if texture == wall_texture:
                    tile_x = x
                    tile_y = z
                    tile_z = -y
                else:  # floor texture
                    tile_x = x
                    tile_y = -1
                    tile_z = -y

                tile_model_matrix = glm.translate(glm.mat4(1.0), glm.vec3(tile_x, tile_y, tile_z)) 

                glUniformMatrix4fv(glGetUniformLocation(shader.program, "model"), 1, GL_FALSE, glm.value_ptr(tile_model_matrix))
                glDrawArrays(GL_QUADS, 0, len(vertices))  # Draw the entire batch at once

if __name__ == "__main__":
    game = SetupGame()
    
    while True:  
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        handle_input(game.camera) 
        draw(game.shader, game.camera, game.vertices, game.room, game.wall_texture, game.floor_texture) 
        pygame.display.flip()  
        pygame.time.wait(10) 