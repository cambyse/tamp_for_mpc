#Include = '../../data/baxter_model/baxter.ors'
#Delete shape visual

#Include = '../../data/reba-human/human.ors'
Include = 'data/man_model.ors'

### objs
#target
body target { type=1 rel=<T t(0 0.0 0.05)> size=[0 0 0 .05] color=[0 1 0] }

#right
body container_0 { X=<T t(-.45 -0.6 0.93) d(-40 0 0 1)> contact }
shape container_0_front(container_0)  { type=9 rel=<T t(0  0.15 0.15)> size=[0.3 0.01 0.3 0.005]   color=[1 0 0] contact}
shape container_0_left(container_0)   { type=9 rel=<T t(.15  0.0  0.15)> size=[0.01 0.3 0.3 0.005] color=[1 0 0] contact}
shape container_0_right(container_0)  { type=9 rel=<T t(-.15 0.0  0.15)> size=[0.01 0.3 0.3 0.005] color=[1 0 0] contact}
shape container_0_bottom(container_0) { type=9 rel=<T t(0  0.0 0)> size=[0.3 0.3 0.01 0.005]     color=[1 0 0] contact}

#middle
body container_1 { X=<T t(.2 -1.05 0.93)> contact }
shape container_1_front(container_1)  { type=9 rel=<T t(0  0.15 0.2)> size=[0.3 0.01 0.4 0.005]   color=[1 0 0] contact}
shape container_1_left(container_1)   { type=9 rel=<T t(.15  0.0  0.2)> size=[0.01 0.3 0.4 0.005] color=[1 0 0] contact}
shape container_1_right(container_1)  { type=9 rel=<T t(-.15 0.0  0.2)> size=[0.01 0.3 0.4 0.005] color=[1 0 0] contact}
shape container_1_bottom(container_1) { type=9 rel=<T t(0  0.0 0)> size=[0.3 0.3 0.01 0.005]     color=[1 0 0] contact}

#Edit /human/base { X=<T t(1.5 0. 1.1) d(-180 0 0 1)> }

#body tableC{ type=9, X=<T t(.7 0 .8)>, size=[1. .8 .04 .02], color=[.3 .3 .5] fixed, contact }
#body tableL{ type=9, X=<T t(.2 .7 .8)>, size=[2. .6 .04 .02], color=[.3 .5 .3] fixed, contact }
body tableC{ type=9, X=<T t(0 -.7 0.9)>, size=[2. .8 .04 .01], color=[.3 .5 .3] }

## GRASP references

#body humanGraspRefR { type=5 size=[.1 0 0 0] color=[1 0 0] }
#body humanGraspRefL { type=5 size=[.1 0 0 0] color=[1 0 0] }
#joint humanGraspJointR(handR humanGraspRefR){ A=<T t(0 0 -.1)> type=11 ctrl_H=1e-4 }
#joint humanGraspJointL(handL humanGraspRefL){ A=<T t(0 0 -.1)> type=11 ctrl_H=1e-4 }
shape humanR (handR){ type=5 size=[.1 0 0 0] color=[1 1 0] rel=<T t(0 0 -.05) d(90 0 0 1)> }
shape humanL (handL){ type=5 size=[.1 0 0 0] color=[1 1 0] rel=<T t(0 0 -.05) d(90 0 0 1)> }

## Joints
joint (tableC container_0) { from=<T t(-0.45 0.1 0.02) d(-40 0 0 1)> type=10 } #from=<T t(0 0 0) t(0 0 0)> to=<T >
joint (tableC container_1)  { from=<T t(0.2 -0.35 0.02)> type=10 } #from=<T t(0 0 0) t(0 0 0)> to=<T >
        
#joint (tableC container_1)  { from=<T t(0.2 -0.3 0.02)> type=10 } #from=<T t(0 0 0) t(0 0 0)> to=<T >

BELIEF_START_STATE{ 
	{ 
          joint (container_0 target) { type=10 }
	}
	{ 
	  joint (container_1 target) { type=10 }
        }
 }
}
