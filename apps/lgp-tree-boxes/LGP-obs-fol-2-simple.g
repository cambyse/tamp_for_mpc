Include = 'data/keywords.g'
#Include = 'LGP-obs-kin-2.g'

FOL_World{
  hasWait=false
  gamma = 1.
  stepCost = 1.
  timeCost = 0.
}

## activities
get_sight
take_view

## basic predicates
agent
location
object

in_sight   # location X is visible
view_taken # object recognition has been triggered for checking if the object X is at location Y
viewed	   # object X has been viewed at location Y
at         # object X is at location Y

now_in_sight # getting sight of location X

# keyword
NOT_OBSERVABLE

## constants
target_location_0
target_location_1
target

## initial state
START_STATE { 
{(object target) (location target_location_0) (location target_location_1) }

BELIEF_START_STATE{ 
{
(at target target_location_0)
()=0.2
}
{
(at target target_location_1)
()=0.8
}
}
### RULES 

#termination rule
#Rule {
#  { (view_taken target target_location_1) (view_taken target target_location_2)} # 
#  { (QUIT) }
#}

Rule {
  { (viewed target target_location_0)} # 
  { (QUIT) }
}

Rule {
  { (viewed target target_location_1)} # 
  { (QUIT) }
}

### Reward
REWARD {
#  tree{
#    leaf{ { (grasped handR screwdriverHandle) }, r=10. }
#    weight=1.
#  }
}

### Activities definitions
# Get sight
DecisionRule get_sight {
  X
  { (INFEASIBLE get_sight X)! (location X) }
  { (now_in_sight X)  komoGetSight(X)=1. }
}

# Take a view
DecisionRule take_view {
  X, Y
  { (object X) (in_sight Y) }
  { (view_taken X Y) komoTakeView(X Y) }
}

### Rules
# Object seen = Observation model
Rule {
  X, Y
  { (object X) (location Y) (NOT_OBSERVABLE at X Y) (in_sight Y) (view_taken X Y) }
  { (viewed X Y)  (at X Y) (NOT_OBSERVABLE at X Y)!}
}

# remove old in sights
Rule {
  X, Y
  { (now_in_sight X) (in_sight Y) }
  { (in_sight Y)! }
}

# transform now in sight in in sight
Rule {
  X
  { (now_in_sight X) }
  { (now_in_sight X)! (in_sight X) }
}

