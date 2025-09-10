
import math
import json
import random
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import pygame

# ==============================
# NeuroRacer (Pygame edition)
# - Single file
# - Checkpoint gates stop spin farming
# - GA + tiny NN, sensors, simple physics
# Controls:
#   Space: run/pause
#   N: next generation
#   R: reset
#   E: export best genome to neuroracer_best.json
#   L: load neuroracer_best.json (seed pop with mutants)
# ==============================

WIDTH, HEIGHT = 1000, 660
FPS = 60

# --- Track Definition (centerline polyline + width) ---
def make_track():
    w, h = 980, 620
    margin = 90
    pts = [
        (margin, h/2),
        (margin, margin),
        (w/2 - 120, margin),
        (w/2 + 160, margin + 20),
        (w - margin, margin + 40),
        (w - margin, h/2 - 60),
        (w - margin - 60, h - margin),
        (w/2 + 60, h - margin - 20),
        (w/2 - 220, h - margin - 40),
        (margin, h - margin),
        (margin, h/2),
    ]
    width = 120
    return {"pts": pts, "width": width, "w": w, "h": h}

TRACK = make_track()

# --- Math helpers ---
def clamp(v,a,b): return max(a, min(b, v))
def seg_len(p,q): return math.hypot(q[0]-p[0], q[1]-p[1])

def point_to_seg(px, py, ax, ay, bx, by):
    vx, vy = bx-ax, by-ay
    wx, wy = px-ax, py-ay
    L2 = vx*vx + vy*vy or 1e-9
    t = clamp((wx*vx + wy*vy) / L2, 0.0, 1.0)
    projx = ax + t*vx
    projy = ay + t*vy
    d = math.hypot(px - projx, py - projy)
    return d, t, projx, projy

# Precompute total track length
TOTAL_TRACK_LEN = sum(seg_len(TRACK["pts"][i], TRACK["pts"][i+1]) for i in range(len(TRACK["pts"])-1))

# distance to centerline + progress (0..1)
def track_distance_and_progress(x, y):
    best = {"d": 1e9, "t": 0.0, "seg": 0, "projx":0.0, "projy":0.0, "accLen":0.0}
    acc = 0.0
    for i in range(len(TRACK["pts"])-1):
        p = TRACK["pts"][i]; q = TRACK["pts"][i+1]
        d,t,projx,projy = point_to_seg(x,y,p[0],p[1],q[0],q[1])
        if d < best["d"]:
            best = {"d": d, "t": t, "seg": i, "projx": projx, "projy": projy, "accLen": acc + seg_len(p,q)*t}
        acc += seg_len(p,q)
    progress = best["accLen"] / (TOTAL_TRACK_LEN or 1.0)
    edge_dist = TRACK["width"]/2 - best["d"]
    return edge_dist, best["seg"], best["t"], progress

# Raycast sensor: walk until outside
def sense_distance(x,y,ang,maxDist=220,step=6):
    d=0.0
    while d < maxDist:
        sx = x + math.cos(ang)*d
        sy = y + math.sin(ang)*d
        edgeDist, *_ = track_distance_and_progress(sx, sy)
        if edgeDist < 0: return d
        d += step
    return maxDist

def cross2(ax,ay,bx,by): return ax*by - ay*bx
def side_of(a,b,p): return cross2(b[0]-a[0], b[1]-a[1], p[0]-a[0], p[1]-a[1])

# --- Checkpoints: one gate per centerline segment (perpendicular across the road) ---
def build_gates():
    gates = []
    pts = TRACK["pts"]
    half = TRACK["width"] / 2
    for i in range(len(pts)-1):
        p, q = pts[i], pts[i+1]
        mx, my = (p[0]+q[0])/2, (p[1]+q[1])/2
        dx, dy = q[0]-p[0], q[1]-p[1]
        L = math.hypot(dx, dy) or 1.0
        ux, uy = dx/L, dy/L          # along track
        nx, ny = -uy, ux             # left normal
        ax, ay = mx - nx*half, my - ny*half
        bx, by = mx + nx*half, my + ny*half
        gates.append({"a":(ax,ay), "b":(bx,by), "dir":(ux,uy)})
    return gates

GATES = build_gates()

# --- Neural Net / Genome ---
HIDDEN = 8
SENSORS = 5

def randn():
    # Box-Muller
    import random, math
    u = 0
    v = 0
    while u == 0: u = random.random()
    while v == 0: v = random.random()
    return math.sqrt(-2.0*math.log(u)) * math.cos(2.0*math.pi*v)

def genome_size(sensor_count=SENSORS):
    input_sz = sensor_count + 1 # +speed
    W1 = (input_sz + 1) * HIDDEN
    W2 = (HIDDEN + 1) * 2
    return W1 + W2

def random_genome(sensor_count=SENSORS):
    n = genome_size(sensor_count)
    return [randn()*0.7 for _ in range(n)]

def forward(genome, inputs, sensor_count=SENSORS):
    input_sz = sensor_count + 1
    W1_size = (input_sz + 1) * HIDDEN
    W1 = genome[:W1_size]
    W2 = genome[W1_size:]
    # inputs + bias
    xin = inputs + [1.0]
    # hidden
    h = []
    for i in range(HIDDEN):
        s = 0.0
        for j in range(input_sz+1):
            s += xin[j] * W1[i*(input_sz+1) + j]
        h.append(math.tanh(s))
    hout = h + [1.0]
    out = []
    for i in range(2):
        s = 0.0
        for j in range(HIDDEN+1):
            s += hout[j] * W2[i*(HIDDEN+1) + j]
        out.append(math.tanh(s))
    return out  # [steer(-1..1), throttle(-1..1)]

def mutate(g, rate):
    ng = g[:]
    for i in range(len(ng)):
        if random.random() < rate:
            ng[i] += randn()*0.4
    return ng

def crossover(a,b):
    n = len(a); cut = random.randrange(n)
    return a[:cut] + b[cut:]

# --- Car ---
@dataclass
class Car:
    x: float
    y: float
    angle: float
    vx: float = 0.0
    vy: float = 0.0
    px: float = 0.0
    py: float = 0.0
    alive: bool = True
    genome: List[float] = field(default_factory=list)
    fitness: float = 0.0
    best_progress: float = 0.0
    lap: int = 0
    nextSeg: int = 0
    color: Tuple[int,int,int] = (255,255,255)
    gateIndex: int = 0
    lastGateTick: int = 0

# --- Simulation Parameters ---
POPULATION = 60
MUTATION = 0.12
ELITES = 4
SENSOR_COUNT = SENSORS
MAX_SIM_TIME = 50.0  # seconds
SPEED_LIMIT = 3.6    # px/frame

# --- Population ---
cars: List[Car] = []
best_genome: Optional[List[float]] = None
running = False
generation = 1
sim_time = 0.0

def spawn_population(genomes: Optional[List[List[float]]] = None):
    global cars
    cars = []
    start = TRACK["pts"][0]
    heading = math.atan2(TRACK["pts"][1][1]-start[1], TRACK["pts"][1][0]-start[0])
    pop = genomes if genomes is not None else [random_genome(SENSOR_COUNT) for _ in range(POPULATION)]
    for i,g in enumerate(pop):
        jitter = lambda: (random.random()-0.5)*8
        x0, y0 = start[0]+jitter(), start[1]+jitter()
        hue = (i*47) % 360
        # Simple HSL-ish to RGB approximation for variety
        color = pygame.Color(0)
        color.hsla = (hue, 80, 55, 100)
        cars.append(Car(
            x=x0, y=y0, angle=heading, px=x0, py=y0,
            genome=g, color=(color.r, color.g, color.b),
        ))

def tournament_pick(sorted_cars, k=5):
    best_idx = random.randrange(len(sorted_cars))
    for _ in range(1,k):
        j = random.randrange(len(sorted_cars))
        if sorted_cars[j].fitness > sorted_cars[best_idx].fitness:
            best_idx = j
    return sorted_cars[best_idx].genome

def evolve():
    global cars, best_genome, generation, sim_time
    sorted_cars = sorted(cars, key=lambda c: c.fitness, reverse=True)
    best_genome = sorted_cars[0].genome[:]
    # save to disk
    try:
        with open("neuroracer_best.json","w") as f:
            json.dump(best_genome,f)
    except Exception:
        pass
    survivors = [c.genome[:] for c in sorted_cars[:max(1,ELITES)]]
    next_pop = survivors[:]
    while len(next_pop) < POPULATION:
        p1 = tournament_pick(sorted_cars)
        p2 = tournament_pick(sorted_cars)
        child = crossover(p1,p2)
        child = mutate(child, MUTATION)
        next_pop.append(child)
    spawn_population(next_pop)
    generation += 1
    sim_time = 0.0

def reset_all():
    global generation, sim_time, best_genome
    generation = 1
    sim_time = 0.0
    best_genome = None
    spawn_population()

# --- Step ---
def step(dt):
    global sim_time
    alive = 0
    best_fit = 0.0
    sum_fit = 0.0
    top_lap = 0

    for c in cars:
        if not c.alive:
            sum_fit += c.fitness
            continue

        c.lastGateTick += 1

        # Sensors
        angles = [-0.8, -0.35, 0.0, 0.35, 0.8]
        sensors = []
        for i in range(SENSOR_COUNT):
            rel = angles[int(i*(len(angles)-1)/max(1,(SENSOR_COUNT-1)))] if SENSOR_COUNT>1 else 0.0
            d = sense_distance(c.x, c.y, c.angle + rel)
            sensors.append(d/220.0)
        speed = math.hypot(c.vx, c.vy)
        inputs = sensors + [speed / SPEED_LIMIT]
        steer_raw, throttle_raw = forward(c.genome, inputs, SENSOR_COUNT)
        steer = clamp(steer_raw, -1.0, 1.0)
        throttle = clamp((throttle_raw+1)/2, 0.0, 1.0)

        # Dynamics
        maxSteer = 0.07
        c.angle += steer * maxSteer
        acc = 0.12 * throttle
        c.vx += math.cos(c.angle) * acc
        c.vy += math.sin(c.angle) * acc

        # Drag + speed cap
        c.vx *= 0.98; c.vy *= 0.98
        spd = math.hypot(c.vx, c.vy)
        if spd > SPEED_LIMIT:
            scale = SPEED_LIMIT / (spd + 1e-6)
            c.vx *= scale; c.vy *= scale

        # Integrate
        ox, oy = c.x, c.y
        c.px, c.py = ox, oy
        c.x += c.vx; c.y += c.vy

        # Bounds check
        edgeDist, segIndex, tOnSeg, progress = track_distance_and_progress(c.x, c.y)
        if edgeDist < -2:  # off track
            c.alive = False

        # Gate crossing
        gk = c.gateIndex % len(GATES)
        gate = GATES[gk]
        s1 = side_of(gate["a"], gate["b"], (ox, oy))
        s2 = side_of(gate["a"], gate["b"], (c.x, c.y))
        if s1 * s2 < 0:  # crossed
            mx, my = c.x - ox, c.y - oy
            forwardDot = mx*gate["dir"][0] + my*gate["dir"][1]
            if forwardDot > 0:
                c.fitness += 5.0
                c.gateIndex = (c.gateIndex + 1) % len(GATES)
                c.lastGateTick = 0
                if gk == 0:
                    c.lap += 1
                    c.fitness += 3.0
            else:
                c.fitness -= 6.0
                c.vx *= 0.7; c.vy *= 0.7

        # Anti-stall
        if c.lastGateTick > 60*6 and math.hypot(c.vx, c.vy) < 0.4:
            c.alive = False

        # Legacy small progress reward
        progBonus = progress + c.lap
        if progBonus > c.best_progress:
            c.fitness += 1.2 * (progBonus - c.best_progress)
            c.best_progress = progBonus
        else:
            c.fitness += 0.0006

        # Legacy seg progress
        if c.nextSeg == segIndex:
            c.nextSeg = (segIndex + 1) % (len(TRACK["pts"])-1)
        elif segIndex == 0 and c.nextSeg == 0:
            c.lap += 1; c.nextSeg = 1
            c.fitness += 2.0

        if c.alive: alive += 1
        best_fit = max(best_fit, c.fitness)
        top_lap = max(top_lap, c.lap)
        sum_fit += c.fitness

    sim_time += dt
    # evolve?
    if alive == 0 or sim_time > MAX_SIM_TIME:
        evolve()

# --- Drawing ---
def draw(screen, font_small):
    screen.fill((15,23,42))  # slate-900

    # Track polylines
    # Asphalt layer
    pygame.draw.lines(screen, (31,41,55), False, TRACK["pts"], TRACK["width"]+18)
    # Lane
    pygame.draw.lines(screen, (55,65,81), False, TRACK["pts"], TRACK["width"]+2)
    # Centerline dashed: approximate by short segments
    center_color = (156,163,175)
    dash_len = 10; gap = 12
    for i in range(len(TRACK["pts"])-1):
        p = TRACK["pts"][i]; q = TRACK["pts"][i+1]
        L = seg_len(p,q)
        steps = int(L // (dash_len+gap)) + 1
        for s in range(steps):
            t0 = (s*(dash_len+gap))/L
            t1 = min((s*(dash_len+gap)+dash_len)/L, 1.0)
            x0 = p[0] + (q[0]-p[0])*t0
            y0 = p[1] + (q[1]-p[1])*t0
            x1 = p[0] + (q[0]-p[0])*t1
            y1 = p[1] + (q[1]-p[1])*t1
            pygame.draw.line(screen, center_color, (x0,y0), (x1,y1), 2)

    # Start line (visual)
    p0 = TRACK["pts"][0]; p1 = TRACK["pts"][1]
    ang = math.atan2(p1[1]-p0[1], p1[0]-p0[0]) - math.pi/2
    # Draw as small rect centered at p0 rotated (approx by line here)
    nx = math.cos(ang); ny = math.sin(ang)
    ax = p0[0] - nx*TRACK["width"]/2; ay = p0[1] - ny*TRACK["width"]/2
    bx = p0[0] + nx*TRACK["width"]/2; by = p0[1] + ny*TRACK["width"]/2
    pygame.draw.line(screen, (229,231,235), (ax,ay), (bx,by), 8)

    # Gates faint
    for i,g in enumerate(GATES):
        col = (71,85,105) if i%2==0 else (51,65,85)
        pygame.draw.line(screen, col, g["a"], g["b"], 2)

    # Best car
    best_car = max(cars, key=lambda c: c.fitness) if cars else None

    # Cars
    for c in cars:
        r = 8
        # body rectangle rotated
        # compute corners
        ca = math.cos(c.angle); sa = math.sin(c.angle)
        # rectangle of width 16, height 9.6
        w = r*2; h = r*1.2
        # center at (c.x,c.y); draw as simple rotated box via polygon
        corners = [(-r, -r*0.6), (r, -r*0.6), (r, r*0.6), (-r, r*0.6)]
        world = []
        for (ux,uy) in corners:
            wx = c.x + ux*ca - uy*sa
            wy = c.y + ux*sa + uy*ca
            world.append((wx,wy))
        pygame.draw.polygon(screen, c.color if c.alive else (17,24,39), world)
        # nose
        nose = [(r*0.9, -r*0.4), (r*1.5, -r*0.4), (r*1.5, r*0.4), (r*0.9, r*0.4)]
        nosew = []
        for (ux,uy) in nose:
            wx = c.x + ux*ca - uy*sa
            wy = c.y + ux*sa + uy*ca
            nosew.append((wx,wy))
        pygame.draw.polygon(screen, (249,250,251) if c.alive else (55,65,81), nosew)

    # Sensors for best car + highlight next gates
    if best_car:
        angles = [-0.8, -0.35, 0.0, 0.35, 0.8]
        for i in range(SENSOR_COUNT):
            rel = angles[int(i*(len(angles)-1)/max(1,(SENSOR_COUNT-1)))] if SENSOR_COUNT>1 else 0.0
            d = sense_distance(best_car.x, best_car.y, best_car.angle+rel)
            ex = best_car.x + math.cos(best_car.angle+rel)*d
            ey = best_car.y + math.sin(best_car.angle+rel)*d
            pygame.draw.line(screen, (253,230,138), (best_car.x,best_car.y), (ex,ey), 2)
            pygame.draw.circle(screen, (253,230,138), (int(ex),int(ey)), 3)

        # Highlight next few gates
        for k in range(3):
            gi = (best_car.gateIndex + k) % len(GATES)
            g = GATES[gi]
            col = (245,158,11) if k==0 else (251,191,36)
            pygame.draw.line(screen, col, g["a"], g["b"], 3 if k==0 else 2)

        # HUD
        speed = math.hypot(best_car.vx, best_car.vy)
        lines = [
            f"Best fitness: {best_car.fitness:.2f}   Lap: {best_car.lap}",
            f"Speed: {speed:.2f} px/f   Gate: {best_car.gateIndex}/{len(GATES)}",
        ]
        for i,txt in enumerate(lines):
            surf = font_small.render(txt, True, (243,244,246))
            screen.blit(surf, (14, 14 + i*16))

def main():
    global running, POPULATION, MUTATION, ELITES, SENSOR_COUNT, MAX_SIM_TIME, SPEED_LIMIT, sim_time, generation
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NeuroRacer — Pygame Edition")
    clock = pygame.time.Clock()
    font_small = pygame.font.SysFont("consolas", 14)

    reset_all()

    running = False

    while True:
        dt = clock.tick(FPS) / 1000.0  # seconds
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    running = not running
                elif event.key == pygame.K_n:
                    evolve()
                elif event.key == pygame.K_r:
                    reset_all()
                elif event.key == pygame.K_e:
                    # Export best
                    global best_genome
                    if best_genome is None and cars:
                        cg = max(cars, key=lambda c: c.fitness).genome
                        try:
                            with open("neuroracer_best.json","w") as f:
                                json.dump(cg,f)
                            print("Exported neuroracer_best.json")
                        except Exception as ex:
                            print("Export failed:", ex)
                    else:
                        try:
                            with open("neuroracer_best.json","w") as f:
                                json.dump(best_genome,f)
                            print("Exported neuroracer_best.json")
                        except Exception as ex:
                            print("Export failed:", ex)
                elif event.key == pygame.K_l:
                    # Load best
                    try:
                        with open("neuroracer_best.json","r") as f:
                            g = json.load(f)
                        base = g[:]
                        genomes = [base]
                        while len(genomes) < POPULATION:
                            genomes.append(mutate(base, 0.5))
                        spawn_population(genomes)
                        generation = 1
                        sim_time = 0.0
                        print("Loaded neuroracer_best.json")
                    except Exception as ex:
                        print("Load failed:", ex)

        if running:
            step(1.0/FPS)

        draw(screen, font_small)
        pygame.display.flip()

if __name__ == "__main__":
    main()
