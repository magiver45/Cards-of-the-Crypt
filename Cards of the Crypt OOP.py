import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import glm
try:
    from PIL import Image
except ImportError:
    print("Install Pillow to load images: pip install Pillow")
    exit()

# --- Constants ---
WIN_WIDTH  = 800
WIN_HEIGHT = 600
GRID_SIZE  = 1.0

# --- Camera ---
class Camera:
    def __init__(self, position, target=None, up=glm.vec3(0, 1, 0), rotation_speed=1.0, grid_size=1.0):
        self.position = position
        self.target = target if target else position + glm.vec3(0, 0, -1)
        self.up = up
        self.rotation_speed = rotation_speed 
        self.grid_size = grid_size
        self.target_position = self.position 
        self.calculate_sides()

    def get_view_matrix(self):
        return glm.lookAt(self.position, self.target, self.up)
    
    def calculate_sides(self):
        self.right = glm.normalize(glm.cross(self.target, self.up)) 
        self.left = -self.right

    def strafe(self, direction):
        if direction > 0: 
            return self.right  # Return the right vector for positive direction
        elif direction < 0:
            return self.left   # Return the left vector for negative direction
        else:
            return glm.vec3(0, 0, 0)  # Return a zero vector if direction is 0

    def rotate_left(self):
        rotation_amount = glm.radians(self.rotation_speed)
        self.target = glm.rotate(self.target, rotation_amount, self.up)
        self.calculate_sides()

    def rotate_right(self):
        rotation_amount = glm.radians(self.rotation_speed) 
        self.target = glm.rotate(self.target, rotation_amount, self.up)
        self.calculate_sides()

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
                print(f"Shader compiled successfully (type: {shader_type})")  # Success message

        except RuntimeError as error:
            print(f"Shader Error: {error}")
            return None

        return shader 

    def compile_and_link_program(self, vertex_source, fragment_source):
        # 1. Create vertex shader
        vertex_shader = self.compile_shader(vertex_source, GL_VERTEX_SHADER)

        # 2. Create fragment shader
        fragment_shader = self.compile_shader(fragment_source, GL_FRAGMENT_SHADER)

        # 3. Create shader program, attach shaders, and link
        shader_program = glCreateProgram()
        glAttachShader(shader_program, vertex_shader)
        glAttachShader(shader_program, fragment_shader)
        glLinkProgram(shader_program)

        # 4. Check for linking errors (include the error handling code from earlier)
        success = glGetProgramiv(shader_program, GL_LINK_STATUS)
        if not success:
            error_log = glGetProgramInfoLog(shader_program).decode('utf-8')
            raise RuntimeError(f"Error linking shader program:\n{error_log}")     
        else:
            print("Shader program linked successfully")  # Success message  

        # 5. Delete shaders after linking
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
            image_data = np.array(list(image.getdata()), np.uint8)  # Convert to NumPy format
        except (FileNotFoundError, AttributeError):  # Error handling if image isn't found
            print(f"Error loading texture: {image_path}")
            return 

        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)

        # Set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT) 
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR) 
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR) 

        # Load the image data into the texture
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data) 

        return texture 

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
        self.tiles = [[Tile(None) for _ in range(width)] for _ in range(height)]  # Placeholder tiles
        self.generate_room(wall_texture,floor_texture)  # Generate the room layout

    def generate_room(self, wall_texture, floor_texture):
        # Basic Room Generation (replace this with your desired algorithm) 
        for y in range(self.height):
            for x in range(self.width):
                if x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1:  # Check if at the edges
                    self.tiles[y][x].texture = wall_texture
                    self.tiles[y][x].is_wall = True
                else:
                    self.tiles[y][x].texture = floor_texture

class Player:
    def __init__(self, position):
        self.position = position
        self.health = 100
        #self.texture = Texture("player.png")
        self.rotation = 0  # Current rotation in degrees
        self.target_rotation = 0  # Target rotation to rotate towards

    def update(self, delta_time): 
        # ... other player update logic
        smooth_speed = 0.5 # Adjust as needed
        t = clamp(smooth_speed * delta_time, 0.0, 1.0)  
        self.rotation = glm.mix(self.rotation, self.target_rotation, t)

class Enemy:
    def __init__(self, position):
        self.position = position
        self.health = 50
        #self.texture = Texture("enemy.png")

# --- Initialization ---
def init_game():
    pygame.init()
    display = (WIN_WIDTH, WIN_HEIGHT)

    # Request a core profile context  
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    #glEnable(GL_DEPTH_TEST) 
    print(glGetString(GL_VERSION))

    print("Creating shader...")
    vertex_shader_source = Shader.read_file("vertex_shader.vs")
    fragment_shader_source = Shader.read_file("fragment_shader.frag")
    shader = Shader(vertex_shader_source, fragment_shader_source)
    print("Shader created:", shader)
    

    print("Loading textures...")  
    wall_texture = Texture("wall_texture.png")
    floor_texture = Texture("floor_texture.png")
    # ... load player, enemy textures ...
    print("Textures loaded.") 

    print("Creating camera...")
    camera = Camera(glm.vec3(0, 0, 3))
    print("Camera created:", camera)
    print("Camera Position:", camera.position)
    print("Camera Target:", camera.target)

    print("Generating room...") 
    room = DungeonRoom(10, 10, wall_texture, floor_texture)
    print("Room generated.") 

    print("Creating player...")
    player = Player(glm.vec3(5, 5, 0)) 
    print("Player created:", player)

    enemies = [Enemy(glm.vec3(2, 2, 0))] 

    return shader, camera, room, player, enemies

# --- Game Loop Functions ---
def handle_input(camera,player):
    keys = pygame.key.get_pressed()

    movement_speed = 1 * camera.grid_size
    rotation_speed = 5

    if keys[K_w]:  # Forward
        camera.target_position += camera.target * movement_speed
        player.position += camera.target * movement_speed  # Move player forward
        camera.target_position = glm.vec3(  
            round(camera.target_position.x / camera.grid_size) * camera.grid_size,
            round(camera.target_position.y / camera.grid_size) * camera.grid_size,
            round(camera.target_position.z / camera.grid_size) * camera.grid_size)
        
        print("Camera Position:", camera.position)
        print("Camera Target:", camera.target)
        

    if keys[K_s]:  # Backward
        camera.target_position -= camera.target * movement_speed  # Subtract for backward movement
        player.position -= camera.target * movement_speed  # Move player backward
        camera.target_position = glm.vec3(
            round(camera.target_position.x / camera.grid_size) * camera.grid_size,
            round(camera.target_position.y / camera.grid_size) * camera.grid_size,
            round(camera.target_position.z / camera.grid_size) * camera.grid_size)
        
        print("Camera Position:", camera.position)
        print("Camera Target:", camera.target)

    if keys[K_a]:  # Left
        camera.target_position -= glm.cross(camera.target, camera.up) * movement_speed  # Use cross product for strafing
        player.position -= camera.strafe(1) * movement_speed  # Move player left
        camera.target_position = glm.vec3(
            round(camera.target_position.x / camera.grid_size) * camera.grid_size,
            round(camera.target_position.y / camera.grid_size) * camera.grid_size,
            round(camera.target_position.z / camera.grid_size) * camera.grid_size)
        
        print("Camera Position:", camera.position)
        print("Camera Target:", camera.target)

    if keys[K_d]:  # Right
        camera.target_position += glm.cross(camera.target, camera.up) * movement_speed  # Opposite direction for right movement
        player.position += camera.strafe(1) * movement_speed  # Move player right
        camera.target_position = glm.vec3(
            round(camera.target_position.x / camera.grid_size) * camera.grid_size,
            round(camera.target_position.y / camera.grid_size) * camera.grid_size,
            round(camera.target_position.z / camera.grid_size) * camera.grid_size)
        
        print("Camera Position:", camera.position)
        print("Camera Target:", camera.target)

    if keys[K_q]:  # Turn left
        camera.rotate_left()
        player.target_rotation -= rotation_speed

        print("Camera Position:", camera.position)
        print("Camera Target:", camera.target)
    if keys[K_e]:  # Turn right
        camera.rotate_right()
        player.target_rotation += rotation_speed

        print("Camera Position:", camera.position)
        print("Camera Target:", camera.target)

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

def update(delta_time, camera, player, enemies):
    handle_input(camera, player)

    smooth_speed = 0.5 
    t = clamp(smooth_speed * delta_time, 0.0, 1.0)  
    camera.position = glm.mix(camera.position, camera.target_position, glm.smoothstep(0.0, 1.0, t))
    
    player.update(delta_time) # Update the player

    # Very basic enemy movement 
    for enemy in enemies:
        enemy.position += glm.vec3(0.05 * GRID_SIZE, 0, 0)  

    # ... (Other game logic: camera updates, collision detection, AI) ...

def draw(shader, camera, room, player, enemies):
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    shader.use() 

    # --- Texture Setup (if using multiple textures) ---
    texture_location = glGetUniformLocation(shader.program, "ourTexture") 
    glUniform1i(texture_location, 0)  
    glActiveTexture(GL_TEXTURE0)

    # --- Model, View, Projection Setup ---
    view_matrix = camera.get_view_matrix()
    projection_matrix = glm.perspective(glm.radians(45.0), (WIN_WIDTH/WIN_HEIGHT), 0.1, 100.0)

    # Send view and projection matrices as uniforms to the shader
    view_location = glGetUniformLocation(shader.program, "view") 
    glUniformMatrix4fv(view_location, 1, GL_FALSE, glm.value_ptr(view_matrix)) 
    projection_location = glGetUniformLocation(shader.program, "projection")  
    glUniformMatrix4fv(projection_location, 1, GL_FALSE, glm.value_ptr(projection_matrix)) 

    # --- Render the Room ---
    for y in range(room.height):
        for x in range(room.width):
            tile = room.tiles[y][x]
            model_matrix = glm.translate(glm.mat4(1.0), glm.vec3(x * GRID_SIZE, y * GRID_SIZE, 0))
            
            # Send the model matrix uniform to the shader
            model_location = glGetUniformLocation(shader.program, "model")
            glUniformMatrix4fv(model_location, 1, GL_FALSE, glm.value_ptr(model_matrix))

            tile.texture.bind()
            glDrawArrays(GL_QUADS, 0, 4)  # Assuming a quad represents a tile

    # --- Render the Player ---
    model_matrix = glm.translate(glm.mat4(1.0), player.position) * glm.rotate(glm.mat4(1.0), player.rotation, glm.vec3(0, 1, 0))

    model_location = glGetUniformLocation(shader.program, "model") 
    glUniformMatrix4fv(model_location, 1, GL_FALSE, glm.value_ptr(model_matrix)) 
    #player.texture.bind() 
    glDrawArrays(GL_QUADS, 0, 4)  # Assuming a quad represents the player

    # --- Render Enemies (similar to rendering the player )---
    for enemy in enemies:
        model_matrix = glm.translate(glm.mat4(1.0), enemy.position)
        glUniformMatrix4fv(model_location, 1, GL_FALSE, glm.value_ptr(model_matrix)) 
        #enemy.texture.bind()
        glDrawArrays(GL_QUADS, 0, 4)  # Assuming a quad represents an enemy

    pygame.display.flip()

# --- Main ---
if __name__ == "__main__":
    shader, camera, room, player, enemies= init_game()
    previous_time = pygame.time.get_ticks()

    while True:  
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        current_time = pygame.time.get_ticks()
        delta_time = (current_time - previous_time) / 1000.0 
        previous_time = current_time

        update(delta_time, camera, player, enemies) 
        draw(shader, camera, room, player, enemies) 
        pygame.display.flip()  
        pygame.time.wait(10) 