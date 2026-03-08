import serial

def calculate_checksum(values):
    """
    FF FE (Header)는 제외
    LEN/CHK/Mode/Data 계산 규칙:
    CHK = ~(ID + LEN + MODE + DATA...) & 0xFF
    """
    return (~sum(values)) & 0xFF

def build_packet(id_, mode, data_bytes):
    """
    패킷 구조: [FF FE] [ID] [LEN] [CHK] [MODE] [DATA...]
    LEN = (MODE+DATA 길이) + 1(CHK 포함)
    """
    payload = [mode] + list(data_bytes)
    length = len(payload) + 1  # +1 for CHK
    checksum = calculate_checksum([id_, length] + payload)
    return bytes([0xFF, 0xFE, id_, length, checksum] + payload)

# --- 명령 ---
def send_control_on(ser, motor_id):
    pkt = build_packet(motor_id, 0x0A, [0x00])
    ser.write(pkt)
    print(f"[TX] CONTROL ON: {pkt.hex()}")

def send_speed_ctrl_params(ser, motor_id, kp=0xFE, ki=0xFE, kd=0x00, current=0x20):
    pkt = build_packet(motor_id, 0x05, [kp, ki, kd, current])
    ser.write(pkt)
    print(f"[TX] SPEED CTRL PARAMS: {pkt.hex()}")

def send_position_ctrl_params(ser, motor_id, kp=0xFE, ki=0xFE, kd=0x00, current=0x20):
    pkt = build_packet(motor_id, 0x04, [kp, ki, kd, current])
    ser.write(pkt)
    print(f"[TX] POSITION CTRL PARAMS: {pkt.hex()}")

def set_position_mode(ser, motor_id, mode_value=0x00):
    pkt = build_packet(motor_id, 0x0B, [mode_value])
    ser.write(pkt)
    print(f"[TX] POSITION MODE SET: {pkt.hex()}")

def send_velocity_mode(ser, motor_id, direction, speed_rpm):
    sp_val = int(speed_rpm / 0.1)
    accel_time = 10  # 1.0s
    data = [direction, (sp_val >> 8) & 0xFF, sp_val & 0xFF, accel_time]
    pkt = build_packet(motor_id, 0x03, data)
    ser.write(pkt)
    print(f"[TX] SPEED CMD: {pkt.hex()}")

def send_position_mode(ser, motor_id, direction, position_deg, speed_rpm):
    pos_val = int(position_deg / 0.01)
    sp_val = int(speed_rpm / 0.1)
    data = [
        direction,
        (pos_val >> 8) & 0xFF, pos_val & 0xFF,
        (sp_val >> 8) & 0xFF, sp_val & 0xFF
    ]
    pkt = build_packet(motor_id, 0x01, data)
    ser.write(pkt)
    print(f"[TX] POSITION CMD: {pkt.hex()}")

def request_encoder_feedback(ser, motor_id):
    pkt = build_packet(motor_id, 0xA9, [])
    ser.write(pkt)
    print(f"[TX] ENCODER REQ: {pkt.hex()}")

# --- 응답 파싱 ---
def read_response(ser):
    if ser.in_waiting < 5:
        return None
    raw = ser.read(ser.in_waiting)
    print(f"[DEBUG] RAW: {raw.hex()}")
    if len(raw) < 5 or raw[:2] != b'\xFF\xFE':
        return None

    id_ = raw[2]
    length = raw[3]
    if len(raw) < 4 + length:
        return None
    chk = raw[4]
    payload = list(raw[5:4 + length])
    if chk != calculate_checksum([id_, length] + payload):
        print("[DEBUG] CHK mismatch")
        return None
    return {"id": id_, "mode": payload[0], "data": payload[1:]}

def parse_encoder(data):
    if len(data) < 3:
        return None
    direction = data[0]
    pos = (data[1] << 8) | data[2]
    angle = pos * 0.01
    return {"direction": direction, "angle": angle}

