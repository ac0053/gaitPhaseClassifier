import numpy as np
import mujoco

def leg_height_con(theta,site_name,leg,joint_id,desired_height,posture_scale_factor,data,model,body):
    ceq=np.zeros(3)

    data.qpos[joint_id]=theta
    mujoco.mj_forward(model, data)
    # calculate leg height based on height of leg from origin and spatial coordinates (x and z) of leg based on ThC
    # joint and tarsus tip
    ee_pos=data.body(site_name).xpos
    body_pos = data.body(body).xpos
    if leg=="RF" or leg=="LF":
        top_leg = data.body(leg+"_ThCBridge").xpos
    else:
        top_leg=data.body(leg+"_Coxa").xpos

    #leg_height=body_pos[2]-ee_pos[2]
    #leg_height = top_leg[2]-ee_pos[2]

    # x1, z1 are spatial coordinates of ThC joint
    # xf, zf are spatial coordinates of tarsus tip
    # height is current height of leg from origin
    # Ensure that foot z pos creates desired height
    ceq[0] = ee_pos[2]-desired_height

    #print(leg_height-desired_height)


    # if hind leg
    if leg=="RH" or leg=="LH":
        ceq[1] = (ee_pos[1] - top_leg[1]) + 1.3 * posture_scale_factor
        ceq[2] = (ee_pos[0] - top_leg[0]) + 1.5 * posture_scale_factor
    elif leg=="RM" or leg=="LM":
        if posture_scale_factor > .01:
            ceq[1] = (ee_pos[1] - top_leg[1]) + 2 * posture_scale_factor
        ceq[2] = (ee_pos[0] - top_leg[0]) + .1 * posture_scale_factor
    else:
        ceq[1] = (ee_pos[1] - top_leg[1]) + .65 * posture_scale_factor


    ## front legs do have an inequality constraint for x

    return np.array(ceq)
