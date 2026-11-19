const KIND = {
  berry:    { label: 'Berry bush',      unit: 'Berries',    nut: 1,   sea: [1.5, 1.1, 1.3, 0.1], rain: 1,   note: 'Fruits in spring to autumn, barren in winter.' },
  tuber:    { label: 'Wild tubers',     unit: 'Roots',      nut: 1.1, sea: [0.5, 0.6, 0.9, 0.7], rain: 1,   note: 'Dug from the ground, so they last through winter.' },
  nut:      { label: 'Nut fall',        unit: 'Nuts',       nut: 1.3, sea: [0.05, 0.2, 2.4, 0.4], rain: 1,  note: 'Under forest canopy; falls heavily in autumn.' },
  mushroom: { label: 'Mushrooms',       unit: 'Mushrooms',  nut: 0.7, sea: [0.6, 0.2, 1.8, 0.1], rain: 2.2, note: 'Damp shade; flushes after rain, thin fare.' },
  grain:    { label: 'Wild grain',      unit: 'Seed heads', nut: 1,   sea: [0.3, 0.9, 1.9, 0],   rain: 1,   note: 'Open ground; ripens late summer into autumn.' },
  cactus:   { label: 'Prickly pear',    unit: 'Fruit',      nut: 0.9, sea: [0.8, 1.4, 0.8, 0.5], rain: 1.5, note: 'Dry country food that holds through the year.' },
  fruit:    { label: 'Wild fruit tree', unit: 'Fruit',    nut: 1.2, sea: [0.9, 1.3, 1.1, 0.4], rain: 1.3, note: 'Figs and mangoes; heavy in the warm wet months, light in the cold.' },
  cattail:  { label: 'Cattails',        unit: 'Shoots',     nut: 0.9, sea: [1.2, 1, 0.8, 0.4],   rain: 1,   note: 'Water edges; shoots and starchy roots.' },
  lichen:   { label: 'Reindeer lichen', unit: 'Lichen',     nut: 0.6, sea: [0.7, 0.9, 0.7, 0.5], rain: 1,   note: 'Cold ground; poor food but always there.' },
  nightshade: { label: 'Nightshade',    unit: 'Berries',    nut: 1,   sea: [1.2, 1.2, 1.2, 0.1], rain: 1,   tox: 0.6, note: 'Glossy dark berries. Poisonous.' },
  toadstool:  { label: 'Toadstools',    unit: 'Mushrooms',  nut: 0.5, sea: [0.6, 0.3, 1.8, 0.1], rain: 2.2, tox: 0.95, note: 'Pale caps in damp shade. Deadly.' },
  apple:    { label: 'Wild apple',      unit: 'Apples',     nut: 1.1, sea: [0.3, 0.7, 2.3, 0.3], rain: 1,   note: 'Orchard-edge trees in temperate woodland. Ripen in autumn and keep into winter.' },
  grape:    { label: 'Wild grape',      unit: 'Grapes',     nut: 0.9, sea: [0.1, 0.8, 2.2, 0.1], rain: 0.8, note: 'Vines on sunny, well-drained slopes. Ripe late summer to autumn.' },
  banana:   { label: 'Wild banana',     unit: 'Bananas',    nut: 1.5, sea: [1.2, 1.3, 1.2, 1],   rain: 1.5, note: 'Tropical wet ground only. Fruits nearly all year, never survives frost.' },
  date:     { label: 'Date palm',       unit: 'Dates',      nut: 1.3, sea: [0.4, 0.9, 2.2, 1],   rain: 0.6, note: 'Desert oases: roots in the water table, crown in the sun. Dates dry sweet and keep.' },
  olive:    { label: 'Wild olive',      unit: 'Olives',     nut: 0.9, sea: [0.1, 0.3, 1.5, 1.8], rain: 0.6, note: 'Dry, rocky, sunny ground. Bitter raw, oily and filling. Ripe late autumn into winter.' },
  allium:   { label: 'Wild onion',      unit: 'Bulbs',      nut: 0.6, sea: [1.5, 0.9, 0.5, 0.1], rain: 1,   note: 'Open meadow and steppe. Bulbs are best in spring and gone by winter.' },
  carrot:   { label: 'Wild carrot',     unit: 'Taproots',   nut: 1,   sea: [0.4, 0.8, 1.3, 0.8], rain: 1,   note: 'Dry grassland and fallow ground. A tough root that stores in the soil.' },
  gourd:    { label: 'Wild gourd',      unit: 'Gourds',     nut: 1.3, sea: [0.2, 1,   1.9, 0.2], rain: 1.2, note: 'Sprawling vine on rich, warm ground. Heavy fruit, hard rind, ripe in autumn.' },
  herb:     { label: 'Healing herb',    unit: 'Leaves',     nut: 0.1, sea: [1.2, 1.2, 1, 0.2],   rain: 1,   heal: true, note: 'Bitter leaves that ease wounds and fever. No food value.' }
};

const A = window.AgonArt;

const INK = A.INK;

function mulberry(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

const SEA = {
  minnow:   { n: 'Minnow',        plan: 'minnow',  c: ['#9fb6b8', '#e4ecea', '#7a9296', '#2f4a50'], len: 14, d: 0, t: [-5, 30], hab: 'Shallows of rivers and lakes', diet: 'Algae, insect larvae', role: 'Prey of everything', note: 'Silver shoals that scatter from any shadow. The base of the water food web.', nut: 0.3 },
  trout:    { n: 'Brook trout',   plan: 'trout',   c: ['#6a7a48', '#e8b068', '#c8643c', '#2e3a22'], len: 24, d: 1, t: [-14, 24], hab: 'Cool, clear, moving water', diet: 'Insects, minnows', role: 'Small hunter', note: 'Spotted flanks and an orange belly. Needs cold water, so it thins out as the climate warms.', nut: 0.9 },
  carp:     { n: 'Common carp',   plan: 'carp',    c: ['#a8883c', '#e0c878', '#8a6a2c', '#4a3a18'], len: 30, d: 1, t: [6, 36], hab: 'Warm, slow, muddy water', diet: 'Weed, snails, grubs', role: 'Grazer', note: 'Deep-bodied bottom feeder with two barbels. Tolerates warm, murky ponds.', nut: 1.2 },
  pike:     { n: 'Pike',          plan: 'pike',    c: ['#5a7a44', '#d8d8a0', '#8a6a3c', '#2a3a20'], len: 36, d: 1, t: [-6, 26], hab: 'Weedy lake edges', diet: 'Fish, frogs, ducklings', role: 'Ambush hunter', note: 'Long jaws, spotted flanks. Hangs motionless in weed and strikes.', nut: 1.4 },
  catfish:  { n: 'Catfish',       plan: 'catfish', c: ['#5a5448', '#b8ac90', '#443e34', '#2a261e'], len: 34, d: 2, t: [14, 40], hab: 'Deep warm pools', diet: 'Anything it can swallow, carrion', role: 'Scavenger', note: 'Whiskered, broad-headed, nocturnal. Lies in the deepest warm pools.', nut: 1.5 },
  eel:      { n: 'River eel',     plan: 'eel',     c: ['#4a4a38', '#a8a070', '#3a3a2a', '#22221a'], len: 38, d: 1, t: [4, 34], hab: 'Muddy banks and slow water', diet: 'Worms, small fish', role: 'Small hunter', note: 'Snake-like and slippery; hides in bank mud by day.', nut: 1 },
  crayfish: { n: 'Crayfish',      plan: 'crayfish',c: ['#a8482c', '#d8886a', '#7a2e1c', '#3a1a10'], len: 16, d: 0, t: [2, 32], hab: 'Stony shallows', diet: 'Detritus, snails', role: 'Scavenger', note: 'Walks the stream bed under stones; claws up when threatened.', nut: 0.6 },
  perch:    { n: 'Yellow perch',  plan: 'perch',   c: ['#b8a040', '#e8e4b0', '#e0702c', '#3a3a20'], len: 22, d: 1, t: [0, 28], hab: 'Weedy lake margins and slow rivers', diet: 'Insect larvae, small fish, crayfish', role: 'Small hunter', note: 'Barred flanks, a spiny first dorsal fin and orange lower fins. Hunts in loose schools and is eaten by pike, herons and otters.', nut: 0.8 },
  salmon:   { n: 'Salmon',        plan: 'salmon',  c: ['#6a7e88', '#e8c8b8', '#c0584a', '#2a3438'], len: 44, d: 2, t: [-4, 16], hab: 'Cold, deep, oxygen-rich water', diet: 'Insects, small fish, crustaceans', role: 'Hunter', note: 'A powerful cold-water swimmer with a hooked jaw in the breeding season. Its flesh is rich and fatty, and its carcasses feed bears, eagles and the whole bank.', nut: 1.6 },
  sturgeon: { n: 'Sturgeon',      plan: 'sturgeon',c: ['#6a6a58', '#d8d4b8', '#8a8a74', '#2c2c24'], len: 70, d: 2, t: [2, 24], hab: 'Deep pools over soft bottoms', diet: 'Snails, mussels, grubs (sucked from the bottom)', role: 'Bottom grazer', note: 'An ancient fish with rows of bony plates, no scales and four feelers on a shovel snout. Grows huge and lives for decades.', nut: 2.4 },
  frog:     { n: 'Bullfrog',      plan: 'frog',    c: ['#5e8a3a', '#d8e0a0', '#3e6224', '#22321a'], len: 18, d: 0, t: [8, 38], hab: 'Reedy shallows and marsh edges', diet: 'Insects, tadpoles, minnows, mice', role: 'Ambush hunter', note: 'Spends the day half-submerged and booms at night. Eats anything that fits in its mouth, and feeds herons, snakes and otters.', nut: 0.5 },
  newt:     { n: 'Newt',          plan: 'newt',    c: ['#5a4a32', '#e08a2c', '#3a2e1e', '#1e1810'], len: 12, d: 0, t: [-2, 28], hab: 'Still, weedy ponds', diet: 'Insect larvae, tadpoles, worms', role: 'Small hunter', note: 'A dark newt with a bright orange belly that warns of poison. Breeds in ponds, spends the winter under logs.', nut: 0.2 },
  mussel:   { n: 'Freshwater mussel', plan: 'mussel', c: ['#3a3a40', '#c8c4bc', '#6a6a72', '#1c1c20'], len: 11, d: 0, t: [0, 30], hab: 'Sandy and gravel beds', diet: 'Filters algae and detritus from the water', role: 'Filter feeder', note: 'Half-buried in the bed, filtering a bathtub of water a day. Clears the water, feeds otters, turtles and colonists with a knife.', nut: 0.6 },
  turtle:   { n: 'Pond turtle',   plan: 'turtle',  c: ['#4a5a38', '#c8b878', '#7a8a4a', '#2a301e'], len: 24, d: 0, t: [8, 36], hab: 'Sunny shallows with banks', diet: 'Weed, snails, carrion', role: 'Grazer', note: 'Domed shell and webbed feet; slow, armoured, hard to kill.', nut: 1.1 }
};

const MAR = {
  sardine:  { n: 'Sardine school', vw: 48, vh: 28, m: 1.6, mv: 2, zone: ['open', 'kelp', 'surf'], t: [8, 28], tl: 'plank', eats: ['zoo', 'phyto'], role: 'Schooling plankton feeder', note: 'A silver ball of thousands of fish that wheels as one. Sardines turn plankton into fish, so almost every hunter at sea depends on them.', nut: 0.5, c: ['#8aa4b8', '#e8eef0', '#5a7488', '#2a4458'] },
  mackerel: { n: 'Mackerel', vw: 44, vh: 16, m: 0.4, mv: 2, zone: ['open', 'surf'], t: [6, 24], tl: 'meso', eats: ['sardine', 'zoo'], role: 'Fast shoaling hunter', note: 'Wavy bars on a blue-green back. Hunts sardines in the surf and is the shore fisher\'s best catch.', nut: 0.9, c: ['#3f7a8a', '#dde8e4', '#2c5a68', '#12303a'] },
  cod:      { n: 'Atlantic cod', vw: 46, vh: 20, m: 0.9, mv: 2, zone: ['kelp', 'open'], t: [-2, 14], tl: 'meso', eats: ['sardine', 'crab', 'urchin', 'starfish'], role: 'Cold-water bottom hunter', note: 'Mottled olive, with a chin barbel and three dorsal fins. Lives over kelp and cold sand and eats anything that fits.', nut: 1.2, c: ['#7a7a48', '#e0dcc0', '#5a5a34', '#2a2a18'] },
  parrotfish: { n: 'Parrotfish', vw: 44, vh: 24, m: 0.6, mv: 2, zone: ['reef'], t: [20, 32], tl: 'herb', eats: ['coral', 'phyto'], role: 'Reef grazer', note: 'Fused beak-like teeth scrape algae off coral. It excretes the fine white sand of tropical beaches.', nut: 0.8, c: ['#2fa89a', '#e86a98', '#4a6ac8', '#16403a'] },
  angelfish: { n: 'Angelfish', vw: 34, vh: 34, m: 0.3, mv: 1.5, zone: ['reef'], t: [22, 32], tl: 'herb', eats: ['coral', 'phyto'], role: 'Reef browser', note: 'A tall, flat disc with long trailing fins. It picks sponges and algae from reef crevices.', nut: 0.4, c: ['#e8c030', '#2a58b8', '#e8c030', '#101830'] },
  clownfish: { n: 'Clownfish', vw: 26, vh: 16, m: 0.1, mv: 1, zone: ['reef'], t: [24, 32], tl: 'meso', eats: ['zoo'], role: 'Anemone dweller', note: 'Immune to the stings of the anemone it lives in. It guards the anemone and eats its leftovers.', nut: 0.2, c: ['#f07820', '#fff', '#f07820', '#101010'] },
  moray:    { n: 'Moray eel', vw: 66, vh: 18, m: 1.5, mv: 0.6, zone: ['reef'], t: [18, 32], tl: 'top', eats: ['clownfish', 'angelfish', 'octopus', 'crab', 'lobster'], role: 'Reef ambusher', note: 'Hides in coral holes with its jaws open, pumping water over its gills. A second set of jaws in the throat drags prey down.', nut: 1.3, c: ['#8a7a2a', '#d8c870', '#6a5a1a', '#2a2208'] },
  stingray: { n: 'Stingray', vw: 52, vh: 48, m: 1.2, mv: 1.5, zone: ['surf', 'open', 'grass'], t: [12, 32], tl: 'scav', eats: ['crab', 'clam', 'lobster', 'starfish'], role: 'Bottom feeder', note: 'Lies buried in sand with only its eyes showing and crushes shellfish with flat plate teeth. The tail barb is for defence.', nut: 1, c: ['#9a8a6a', '#e8e0cc', '#7a6a4a', '#3a3018'] },
  manta:    { n: 'Manta ray', vw: 96, vh: 62, m: 4, mv: 2, zone: ['open', 'deep'], t: [20, 32], tl: 'filt', eats: ['zoo'], role: 'Giant plankton filterer', note: 'Wingspan of four metres or more. Glides with its mouth open and funnels plankton through horn-shaped lobes.', nut: 2, c: ['#2a3a58', '#f4f4f0', '#1a2438', '#0c1220'] },
  turtle:   { n: 'Green sea turtle', vw: 48, vh: 44, m: 1.1, mv: 1.5, zone: ['grass', 'reef', 'surf'], t: [18, 32], tl: 'herb', eats: ['grass', 'kelp'], role: 'Seagrass grazer', note: 'Heart-shaped shell and paddle flippers. Grazes seagrass meadows like a lawnmower and must surface to breathe. Females return to the beach where they hatched.', nut: 1.8, c: ['#6a7a3a', '#d8cc88', '#5a6a2e', '#2a3016'] },
  dugong:   { n: 'Dugong', vw: 70, vh: 30, m: 2.7, mv: 1.5, zone: ['grass'], t: [20, 32], tl: 'herb', eats: ['grass'], role: 'Sea cow', note: 'A slow, whiskered sea mammal that mows seagrass beds and leaves visible trails. Breeds slowly, so it is easily overhunted.', nut: 3, c: ['#8a8a8a', '#b8b8b0', '#6a6a6a', '#2a2a2a'] },
  seahorse: { n: 'Seahorse', vw: 18, vh: 28, m: 0.15, mv: 0.5, zone: ['grass', 'kelp'], t: [14, 30], tl: 'meso', eats: ['zoo'], role: 'Seagrass ambusher', note: 'Grips seagrass with its tail and sucks in tiny crustaceans whole. The male carries the young.', nut: 0.1, c: ['#e8b840', '#f4e0a0', '#c88a20', '#503810'] },
  octopus:  { n: 'Octopus', vw: 44, vh: 44, m: 1.0, mv: 1, zone: ['reef', 'kelp'], t: [8, 30], tl: 'meso', eats: ['crab', 'lobster', 'clam'], role: 'Reef hunter', note: 'Eight arms, three hearts and a boneless body that squeezes through a coin-sized gap. Changes colour to match the rock and sprays ink.', nut: 1.4, c: ['#b8584a', '#e8a08a', '#8a3a30', '#3a1410'] },
  squid:    { n: 'Squid', vw: 52, vh: 20, m: 0.6, mv: 2, zone: ['open', 'deep'], t: [4, 28], tl: 'meso', eats: ['sardine', 'zoo'], role: 'Open-water jet hunter', note: 'Jets through the water by squirting from its mantle. Two long tentacles snatch fish, and it flashes colour at night.', nut: 1, c: ['#d89aa0', '#f4dede', '#a86a74', '#3a1a20'] },
  jellyfish:{ n: 'Moon jelly', vw: 30, vh: 34, m: 0.3, mv: 1, zone: ['surf', 'open'], t: [4, 30], tl: 'plank', eats: ['zoo'], role: 'Drifting plankton catcher', note: 'A translucent bell with four pale rings and trailing stinging tentacles. Drifts with the current and blooms when fish are scarce.', nut: 0.1, c: ['#c8d8f0', '#f0f4ff', '#a8b8e0', '#6a7aa8'] },
  crab:     { n: 'Shore crab', vw: 34, vh: 28, m: 0.15, mv: 1, zone: ['surf', 'kelp', 'reef'], t: [2, 30], tl: 'scav', eats: ['phyto', 'kelp', 'clam'], role: 'Shore scavenger', note: 'Sideways walker that clears carrion and weed from the tideline. A staple of shore foragers because it can be picked up by hand.', nut: 0.7, c: ['#b86038', '#e8a070', '#8a4222', '#3a1c0c'] },
  lobster:  { n: 'Spiny lobster', vw: 54, vh: 24, m: 0.5, mv: 1, zone: ['reef', 'kelp'], t: [8, 30], tl: 'scav', eats: ['urchin', 'clam', 'starfish', 'crab'], role: 'Reef scavenger', note: 'Armoured with long antennae and no big claws. Leaves its crevice at night to eat urchins and shellfish.', nut: 1.3, c: ['#b8402a', '#e89a78', '#8a2e1c', '#3a140c'] },
  urchin:   { n: 'Sea urchin', vw: 24, vh: 24, m: 0.1, mv: 0, zone: ['reef', 'kelp'], t: [0, 32], tl: 'herb', eats: ['kelp', 'phyto', 'coral'], role: 'Reef and kelp grazer', note: 'A ball of spines that rasps algae and kelp. Without otters or lobsters it strips kelp forests bare and leaves urchin barrens.', nut: 0.4, c: ['#4a2a5a', '#8a5a9a', '#2a1438', '#12081c'] },
  starfish: { n: 'Sea star', vw: 28, vh: 28, m: 0.25, mv: 0, zone: ['reef', 'kelp', 'surf'], t: [-2, 32], tl: 'scav', eats: ['clam', 'urchin'], role: 'Shellfish predator', note: 'Pulls open a mussel with its arms and pushes its stomach inside to digest it. Regrows lost arms.', nut: 0.1, c: ['#e8742c', '#f4b078', '#c05418', '#5a2a0c'] },
  clam:     { n: 'Giant clam', vw: 40, vh: 28, m: 1.0, mv: 0, zone: ['reef', 'surf'], t: [22, 32], tl: 'filt', eats: ['phyto', 'zoo'], role: 'Reef filter feeder', note: 'Up to a metre across. Filters the water and farms algae in its colourful mantle. Cannot move and lives for a century.', nut: 1.2, c: ['#9a9a78', '#6a3a9a', '#2a8a9a', '#2a2a1c'] },
  tuna:     { n: 'Yellowfin tuna', vw: 72, vh: 26, m: 1.8, mv: 2, zone: ['open', 'deep'], t: [14, 30], tl: 'top', eats: ['sardine', 'mackerel', 'squid'], role: 'Open-ocean sprinter', note: 'Warm-blooded, torpedo-shaped and built for speed. Swims constantly and follows dolphins to find fish shoals.', nut: 2, c: ['#2a4a7a', '#d8dce0', '#e8c830', '#0e1c30'] },
  barracuda:{ n: 'Great barracuda', vw: 78, vh: 16, m: 1.5, mv: 2, zone: ['reef', 'open', 'surf'], t: [20, 32], tl: 'top', eats: ['sardine', 'mackerel', 'angelfish', 'parrotfish'], role: 'Reef ambusher', note: 'A long silver torpedo with a jutting lower jaw. Hovers motionless and strikes at 40 km/h.', nut: 1.4, c: ['#7a8a98', '#e8ecee', '#5a6a78', '#1a2430'] },
  reefshark:{ n: 'Blacktip reef shark', vw: 84, vh: 32, m: 1.6, mv: 2, zone: ['reef', 'open', 'surf'], t: [20, 32], tl: 'top', eats: ['parrotfish', 'angelfish', 'barracuda', 'stingray', 'mackerel', 'octopus', 'sardine'], role: 'Reef predator', note: 'Black-tipped fins and a pale belly. Patrols the reef edge, shallow enough to see its fin. Rarely dangerous to people unless provoked.', nut: 2, c: ['#7a8890', '#f0f0ec', '#6a7880', '#14181c'] },
  whiteshark:{ n: 'Great white shark', vw: 104, vh: 38, m: 4.5, mv: 2, zone: ['open', 'kelp', 'deep'], t: [8, 24], tl: 'top', eats: ['seal', 'tuna', 'mackerel', 'cod', 'stingray', 'dolphin'], role: 'Apex predator', note: 'Bulky grey above and white below. Ambushes seals from below and breaches in a rush. A bite is rare but terrifying.', nut: 3, c: ['#6a7a86', '#f4f4f2', '#566672', '#161c22'] },
  dolphin:  { n: 'Bottlenose dolphin', vw: 80, vh: 30, m: 2.6, mv: 2, zone: ['open', 'surf', 'reef'], t: [8, 30], tl: 'top', eats: ['sardine', 'mackerel', 'squid', 'cod'], role: 'Pod hunter', note: 'Intelligent pod hunters that herd sardines into tight balls and take turns to feed. Breathe air and sleep with half the brain.', nut: 2.5, c: ['#6a7e8e', '#e4eaee', '#566a7a', '#1a2430'] },
  seal:     { n: 'Harbour seal', vw: 50, vh: 26, m: 1.6, mv: 1.5, zone: ['kelp', 'surf', 'open'], t: [-4, 18], tl: 'top', eats: ['cod', 'sardine', 'squid', 'mackerel'], role: 'Coastal hunter', note: 'Hauls out on rocks and sandbars to rest and pup. Dives for fish and is the favoured prey of white sharks.', nut: 2, c: ['#7a7468', '#cfc8b8', '#5a564c', '#26231e'] },
  otter:    { n: 'Sea otter', vw: 46, vh: 22, m: 1.2, mv: 1.5, zone: ['kelp'], t: [-2, 16], tl: 'top', eats: ['urchin', 'crab', 'clam'], role: 'Kelp guardian', note: 'Floats on its back with a stone to crack shellfish and wraps itself in kelp to sleep. Eating urchins saves the kelp forest.', nut: 1.6, c: ['#6a4a2e', '#c8a478', '#4a3220', '#20140a'] },
  whale:    { n: 'Humpback whale', vw: 160, vh: 74, m: 15, mv: 2, zone: ['deep', 'open'], t: [-2, 28], tl: 'filt', eats: ['sardine', 'zoo'], role: 'Giant filter feeder', note: 'Fifteen metres of knobbled head, huge winglike flippers and a fluke with a unique pattern. Feeds in the cold and breeds in warm shallows, travelling thousands of kilometres.', nut: 6, c: ['#2a3848', '#e8ecf0', '#1c2836', '#0c1018'] }
};

function marSvg(k) {
  const S = MAR[k], [b, bl, fn, mk] = S.c, P = (d, f, w) => '<path d="' + d + '" fill="' + f + '" stroke="' + INK + '" stroke-width="' + (w === undefined ? 1.4 : w) + '" stroke-linejoin="round" stroke-linecap="round"/>', E = (cx, cy, rx, ry, f, w) => '<ellipse cx="' + cx + '" cy="' + cy + '" rx="' + rx + '" ry="' + ry + '" fill="' + f + '" stroke="' + INK + '" stroke-width="' + (w === undefined ? 1.4 : w) + '"/>', EY = (x, y, r) => '<circle cx="' + x + '" cy="' + y + '" r="' + (r || 1.5) + '" fill="#fff" stroke="' + INK + '" stroke-width="0.7"/><circle cx="' + (x + 0.3) + '" cy="' + y + '" r="' + ((r || 1.5) * 0.5) + '" fill="#101010"/>', L = (d, c, w) => '<path d="' + d + '" fill="none" stroke="' + c + '" stroke-width="' + w + '" stroke-linecap="round"/>';
  let o = '';
  if (k === 'sardine') { const f1 = (x, y, r) => '<g transform="translate(' + x + ' ' + y + ') rotate(' + r + ')">' + P('M0 0 L-5 -3 L-5 3Z', fn, 0.8) + E(5, 0, 6, 2.1, b, 0.8) + P('M0 0.6 Q5 2.6 11 0.4 Q5 3.6 0 0.6Z', bl, 0) + '<path d="M0 -0.8 H10" stroke="' + mk + '" stroke-width="0.7"/><circle cx="9.4" cy="-0.4" r="0.7" fill="#101010"/></g>'; o = [[8, 8, 12], [22, 5, -6], [15, 14, 4], [30, 12, -10], [24, 21, 8], [8, 22, -4], [36, 19, 14]].map(([x, y, r]) => f1(x, y, r)).join(''); }
  else if (k === 'mackerel') o = P('M4 8 L-1 3 L-1 13Z', fn) + P('M4 8 Q14 1 30 5 L42 7.4 L42 9 L30 11 Q14 15 4 8Z', b) + P('M8 8.6 Q20 11.4 36 9 L30 11 Q14 15 8 8.6Z', bl, 0) + [10, 14, 18, 22, 26, 30].map((x, i) => '<path d="M' + x + ' ' + (5.4 - (i % 2) * 0.4) + ' q2 2 -0.5 4" fill="none" stroke="' + mk + '" stroke-width="0.9"/>').join('') + P('M16 4 L19 0.6 L22 4.4Z', fn, 1) + P('M12 11.6 L10 15 L15 12.4Z', fn, 1) + EY(37, 6.8, 1.3);
  else if (k === 'cod') o = P('M5 10 L-1 3 L-1 17Z', fn) + P('M4 10 Q10 2 26 3.4 Q38 4 44 9 Q44 12 36 14.6 Q18 18 4 10Z', b) + P('M10 12 Q26 17 40 13 Q18 18.6 10 12Z', bl, 0) + [[12, 7], [18, 6], [24, 9], [29, 6.4], [16, 11], [32, 11]].map(([x, y]) => '<circle cx="' + x + '" cy="' + y + '" r="1.5" fill="' + mk + '" opacity="0.55"/>').join('') + P('M11 3.4 Q14 -0.6 18 3Z', fn, 1) + P('M20 3 Q23 -0.8 27 3.2Z', fn, 1) + P('M29 3.6 Q32 0 35 4.4Z', fn, 1) + L('M41 12 q1 3 -1 4', mk, 1) + EY(38, 7.4, 1.8);
  else if (k === 'parrotfish') o = P('M6 12 L-1 4 L2 12 L-1 20Z', fn) + E(22, 12, 18, 8.6, b) + P('M8 12 Q22 20 38 14 Q22 22 8 12Z', bl, 0) + [[12, 9], [17, 8], [22, 9], [27, 9], [15, 13], [20, 13], [25, 14], [30, 13]].map(([x, y], i) => '<circle cx="' + x + '" cy="' + y + '" r="2.1" fill="' + (i % 2 ? bl : mk) + '" opacity="0.5"/>').join('') + P('M13 4 Q22 -1 32 5Z', fn, 1.1) + P('M34 11 Q41 10 42 13 Q38 15 34 14.6Z', '#f4f0e0', 1.2) + L('M35 12.6 H41', mk, 0.7) + EY(33, 9, 1.8) + P('M18 17 L16 21 L24 18Z', fn, 1);
  else if (k === 'angelfish') o = P('M4 18 L-1 8 L-1 28Z', fn) + P('M14 6 Q18 -2 30 0 Q22 4 20 8Z', fn, 1.2) + P('M14 30 Q18 38 30 36 Q22 32 20 28Z', fn, 1.2) + E(17, 18, 11, 14.4, b) + [[10, 18], [15, 18], [20, 18]].map(([x]) => '<path d="M' + x + ' 6 Q' + (x + 2) + ' 18 ' + x + ' 30" fill="none" stroke="' + bl + '" stroke-width="2.2"/>').join('') + '<path d="M22 10 Q26 18 22 27" fill="none" stroke="#fff" stroke-width="1.2" opacity="0.7"/>' + EY(24, 14, 1.8) + P('M27 20 q2 1 1 3', mk, 1);
  else if (k === 'clownfish') o = P('M5 8 L-1 2 L-1 14Z', fn) + E(13, 8, 10, 5.6, b) + ['M9 2.6 Q7 8 9 13.6', 'M15 2.6 Q13.4 8 15 13.6', 'M20.4 3.6 Q19.6 8 20.4 12.4'].map((d, i) => '<path d="' + d + '" fill="none" stroke="' + INK + '" stroke-width="' + (i === 1 ? 4.2 : 3.6) + '"/><path d="' + d + '" fill="none" stroke="#fff" stroke-width="2.4"/>').join('') + P('M8 3 Q13 -0.6 18 3Z', fn, 1) + EY(20, 6.4, 1.3) + P('M10 13 L8 16 L13 13.6Z', fn, 0.9);
  else if (k === 'moray') o = P('M2 9 Q10 3 20 9 T38 9 T56 8 Q60 8 64 7.4 Q66 9 64 10.8 Q60 11.4 56 12 Q40 16 24 12 T4 12Z', b, 1.3) + [[12, 8], [20, 10], [28, 9], [36, 10], [44, 9.4], [52, 9.6]].map(([x, y]) => '<circle cx="' + x + '" cy="' + y + '" r="2" fill="' + mk + '" opacity="0.5"/>').join('') + P('M54 6 L64 6.4 L65 9 L56 9.4Z', b, 1.2) + P('M56 9.6 L65 10 L63 12.4 L55 11.4Z', bl, 1.1) + '<path d="M58 9.4 l1 1.6 M61 9.6 l1 1.6" stroke="#fff" stroke-width="0.8"/>' + EY(58, 7.4, 1.3) + P('M10 12 Q14 16 18 12Z', fn, 0.9);
  else if (k === 'stingray') o = P('M12 22 L-1 22 L12 24Z', mk, 1) + P('M18 24 L2 24 L0 25.4 L18 25.2Z', mk, 0) + P('M48 24 Q42 4 26 2 Q18 8 14 24 Q18 40 26 46 Q42 44 48 24Z', b, 1.5) + P('M30 16 Q44 14 48 24 Q44 34 30 32Z', bl, 0) + [[28, 14], [32, 22], [26, 30], [36, 28], [22, 22], [34, 14]].map(([x, y]) => '<circle cx="' + x + '" cy="' + y + '" r="1.3" fill="' + mk + '" opacity="0.6"/>').join('') + EY(42, 17, 1.7) + EY(42, 31, 1.7) + L('M10 24 H1', INK, 2.2) + L('M10 24 H1', mk, 1.2);
  else if (k === 'manta') o = P('M66 31 Q60 6 40 2 Q22 14 18 31 Q22 48 40 60 Q60 56 66 31Z', b, 1.6) + P('M60 31 Q70 30 84 31 Q70 34 60 31Z', b, 1.2) + P('M70 31 Q66 36 62 38 M70 31 Q66 26 62 24', b, 0) + P('M28 31 Q50 20 70 31 Q50 42 28 31Z', bl, 0) + '<circle cx="38" cy="26" r="2.4" fill="' + bl + '" opacity="0.9"/><circle cx="38" cy="36" r="2.4" fill="' + bl + '" opacity="0.9"/>' + P('M64 26 Q72 22 78 24 Q72 27 66 28Z', b, 1.1) + P('M64 36 Q72 40 78 38 Q72 35 66 34Z', b, 1.1) + EY(60, 22, 1.5) + EY(60, 40, 1.5) + L('M18 31 H3', INK, 2.4) + L('M18 31 H3', mk, 1.2);
  else if (k === 'turtle') o = P('M10 14 q-8 0 -9 6 q4 3 10 0Z', fn, 1.1) + P('M10 30 q-8 0 -9 -6 q4 -3 10 0Z', fn, 1.1) + P('M32 8 q4 -6 11 -4 q-1 7 -9 9Z', fn, 1.1) + P('M32 36 q4 6 11 4 q-1 -7 -9 -9Z', fn, 1.1) + P('M40 22 q3 -3 7 -1 q1 2 -1 3 q-3 2 -6 1Z', fn, 1.2) + E(23, 22, 16, 13.4, b) + P('M12 16 L19 11 L28 11 L34 17 L34 27 L28 33 L19 33 L12 27Z', bl, 0.9) + '<path d="M17 14 L21 22 L17 30 M27 12 L25 22 L27 32 M12 22 H34 M21 22 H33" fill="none" stroke="' + mk + '" stroke-width="1"/>' + EY(45, 20.6, 1.1);
  else if (k === 'dugong') o = P('M10 14 Q2 6 -1 14 Q2 18 10 14Z', fn, 0) + P('M10 15 Q1 22 -1 14 Q4 20 10 15Z', fn, 1.2) + P('M6 15 Q2 8 -1 8 Q1 15 6 15Z M6 15 Q2 22 -1 22 Q1 15 6 15Z', fn, 1.3) + E(30, 15, 24, 10.4, b) + P('M12 16 Q34 24 52 17 Q34 26 12 16Z', bl, 0) + P('M30 22 L24 29 L34 25Z', fn, 1.2) + P('M48 10 Q60 8 62 14 Q61 20 50 20Z', b, 1.4) + '<path d="M58 17 l2 2 M55 18 l1 2.4 M60 14 l2 0" stroke="' + mk + '" stroke-width="0.9"/>' + EY(54, 11.4, 1.1);
  else if (k === 'seahorse') o = L('M9 18 Q12 26 7 25 Q2 24 3 20', INK, 4.6) + L('M9 18 Q12 26 7 25 Q2 24 3 20', b, 2.8) + P('M10 5 Q15 3 15 10 Q14 15 11 20 Q7 22 7 17 Q8 11 7 8 Q7 5 10 5Z', b, 1.3) + P('M7 5 Q4 4 3 6 L7 7Z', bl, 1) + P('M12 8 Q17 9 16 13Z', fn, 0.9) + P('M7 4 Q4 2 5 -0.4 Q9 1 9 4Z', fn, 1) + EY(9.6, 5.8, 1.1) + [[10, 10], [10, 13], [9.6, 16]].map(([x, y]) => '<path d="M7 ' + y + ' h6" stroke="' + mk + '" stroke-width="0.7" opacity="0.7"/>').join('');
  else if (k === 'octopus') { const arm = (a) => { const r = a * Math.PI / 180, x1 = 22 + Math.cos(r) * 8, y1 = 22 + Math.sin(r) * 8, x2 = 22 + Math.cos(r + 0.5) * 20, y2 = 22 + Math.sin(r + 0.5) * 20, x3 = 22 + Math.cos(r + 0.1) * 21, y3 = 22 + Math.sin(r + 0.1) * 21; return L('M22 22 L' + x1.toFixed(1) + ' ' + y1.toFixed(1) + ' Q' + x2.toFixed(1) + ' ' + y2.toFixed(1) + ' ' + x3.toFixed(1) + ' ' + y3.toFixed(1), INK, 5) + L('M22 22 L' + x1.toFixed(1) + ' ' + y1.toFixed(1) + ' Q' + x2.toFixed(1) + ' ' + y2.toFixed(1) + ' ' + x3.toFixed(1) + ' ' + y3.toFixed(1), b, 3); }; o = [0, 45, 90, 135, 180, 225, 270, 315].map(arm).join('') + E(22, 22, 11, 10, b, 1.5) + E(19, 20, 6, 5, bl, 0) + EY(26, 18, 2) + EY(26, 26, 2) + [[14, 14], [14, 30], [10, 22]].map(([x, y]) => '<circle cx="' + x + '" cy="' + y + '" r="1.2" fill="' + mk + '" opacity="0.6"/>').join(''); }
  else if (k === 'squid') o = P('M10 10 Q2 3 -1 10 Q2 17 10 10Z', fn, 1.1) + L('M30 10 Q42 7 50 6', INK, 3.4) + L('M30 10 Q42 7 50 6', bl, 1.6) + L('M30 11 Q42 13 50 15', INK, 3.4) + L('M30 11 Q42 13 50 15', bl, 1.6) + P('M10 10 Q18 1 30 6 Q36 10 30 14 Q18 19 10 10Z', b, 1.4) + P('M10 10 Q4 1 14 3 Q18 5 17 8Z', fn, 1) + P('M10 10 Q4 19 14 17 Q18 15 17 12Z', fn, 1) + [[18, 8], [22, 11], [26, 8], [20, 13]].map(([x, y]) => '<circle cx="' + x + '" cy="' + y + '" r="1" fill="' + mk + '" opacity="0.6"/>').join('') + EY(30, 8, 1.8);
  else if (k === 'jellyfish') o = [8, 12, 16, 20, 24].map((x, i) => L('M' + x + ' 14 q' + (i % 2 ? 3 : -3) + ' 6 0 10 t0 8', mk, 1.2)).join('') + L('M13 14 q-3 6 0 9 t-1 8 M18 14 q3 7 0 11', bl, 2.2) + P('M3 14 Q3 2 15 2 Q27 2 27 14 Q15 18 3 14Z', b, 1.4) + '<path d="M5 12 Q15 15 25 12" fill="none" stroke="' + bl + '" stroke-width="1" opacity="0.8"/>' + [[10, 8], [20, 8], [10, 11], [20, 11]].map(([x, y]) => '<circle cx="' + x + '" cy="' + y + '" r="2.2" fill="none" stroke="' + mk + '" stroke-width="1" opacity="0.8"/>').join('');
  else if (k === 'crab') { const leg = (x, y, dx, dy) => L('M' + x + ' ' + y + ' l' + dx + ' ' + (dy * 0.4) + ' l' + (dx * 0.6) + ' ' + dy, INK, 3.2) + L('M' + x + ' ' + y + ' l' + dx + ' ' + (dy * 0.4) + ' l' + (dx * 0.6) + ' ' + dy, b, 1.8); o = [[10, 14, -6, 8], [11, 18, -7, 5], [12, 22, -6, 3], [24, 14, 6, 8], [23, 18, 7, 5], [22, 22, 6, 3]].map(([x, y, dx, dy]) => leg(x, y, dx, dy)).join('') + P('M6 10 Q3 2 8 2 Q12 4 10 9Z', fn, 1.3) + P('M28 10 Q31 2 26 2 Q22 4 24 9Z', fn, 1.3) + P('M10 10 Q6 6 6 2', b, 0) + E(17, 16, 11, 8.4, b, 1.5) + '<path d="M10 14 Q17 10 24 14" fill="none" stroke="' + bl + '" stroke-width="1" opacity="0.8"/>' + EY(13, 9, 1.4) + EY(21, 9, 1.4); }
  else if (k === 'lobster') { const an = (d) => L(d, INK, 2.2) + L(d, bl, 0.9); o = an('M44 8 Q52 2 58 -1') + an('M44 12 Q52 14 58 18') + P('M4 12 L-1 6 L3 10 L-1 12 L3 14 L-1 18Z', fn, 1.1) + [[12, 12], [18, 12], [24, 12], [30, 12]].map(([x, y], i) => P('M' + (x - 4) + ' ' + (y - 5 + i * 0.4) + ' Q' + (x + 2) + ' ' + (y - 6) + ' ' + (x + 3.5) + ' ' + (y - 4.6) + ' L' + (x + 3.5) + ' ' + (y + 4.6) + ' Q' + (x + 2) + ' ' + (y + 6) + ' ' + (x - 4) + ' ' + (y + 5 - i * 0.4) + 'Z', i % 2 ? b : bl, 1.1)).join('') + P('M32 6 Q40 2 46 6 Q48 10 46 14 Q40 18 32 14Z', b, 1.4) + '<path d="M34 5 L36 1 M34 15 L36 19 M40 4 L42 0 M40 16 L42 20" stroke="' + INK + '" stroke-width="1.6" stroke-linecap="round"/>' + EY(44, 7.4, 1.3) + EY(44, 12.6, 1.3) + [[38, 10], [30, 10]].map(([x, y]) => '<circle cx="' + x + '" cy="' + y + '" r="1" fill="' + mk + '" opacity="0.5"/>').join(''); }
  else if (k === 'urchin') { let sp = ''; for (let i = 0; i < 20; i++) { const a = i / 20 * Math.PI * 2, r0 = 6, r1 = 11 + (i % 3); sp += '<path d="M' + (12 + Math.cos(a) * r0).toFixed(1) + ' ' + (12 + Math.sin(a) * r0).toFixed(1) + ' L' + (12 + Math.cos(a) * r1).toFixed(1) + ' ' + (12 + Math.sin(a) * r1).toFixed(1) + '" stroke="' + INK + '" stroke-width="2.4" stroke-linecap="round"/><path d="M' + (12 + Math.cos(a) * r0).toFixed(1) + ' ' + (12 + Math.sin(a) * r0).toFixed(1) + ' L' + (12 + Math.cos(a) * r1).toFixed(1) + ' ' + (12 + Math.sin(a) * r1).toFixed(1) + '" stroke="' + (i % 2 ? b : bl) + '" stroke-width="1.1" stroke-linecap="round"/>'; } o = sp + E(12, 12, 5.6, 5.6, mk, 1.3) + '<circle cx="10" cy="10" r="1.4" fill="' + bl + '" opacity="0.6"/>'; }
  else if (k === 'starfish') { const pts = []; for (let i = 0; i < 10; i++) { const a = -Math.PI / 2 + i * Math.PI / 5, r = i % 2 ? 4.6 : 12.4; pts.push((14 + Math.cos(a) * r).toFixed(1) + ' ' + (14 + Math.sin(a) * r).toFixed(1)); } o = P('M' + pts.join(' L') + 'Z', b, 1.5) + [0, 1, 2, 3, 4].map((i) => { const a = -Math.PI / 2 + i * 2 * Math.PI / 5; return L('M14 14 L' + (14 + Math.cos(a) * 9.5).toFixed(1) + ' ' + (14 + Math.sin(a) * 9.5).toFixed(1), bl, 1.2) + '<circle cx="' + (14 + Math.cos(a) * 6).toFixed(1) + '" cy="' + (14 + Math.sin(a) * 6).toFixed(1) + '" r="0.9" fill="' + mk + '"/>'; }).join('') + E(14, 14, 2.4, 2.4, mk, 0.8); }
  else if (k === 'clam') o = P('M4 22 Q-1 8 12 4 Q20 0 28 4 Q41 8 36 22 Q20 28 4 22Z', b, 1.6) + [8, 12, 16, 20, 24, 28, 32].map((x) => '<path d="M' + x + ' 22 Q' + (x - 1) + ' 12 ' + (x + 1) + ' 4" fill="none" stroke="' + mk + '" stroke-width="1" opacity="0.6"/>').join('') + P('M7 20 Q20 14 33 20 Q20 28 7 20Z', bl, 1.2) + '<path d="M10 20 Q20 16 30 20" fill="none" stroke="' + fn + '" stroke-width="2.4"/>' + [[14, 20], [20, 19], [26, 20]].map(([x, y]) => '<circle cx="' + x + '" cy="' + y + '" r="1.4" fill="#e8f088" opacity="0.8"/>').join('');
  else if (k === 'tuna') o = P('M12 13 L-1 2 L4 13 L-1 24Z', fn, 1.4) + P('M10 13 Q18 1 40 3 Q62 4 72 12 Q72 14 62 16 Q40 24 10 13Z', b, 1.5) + P('M16 15 Q40 24 68 14 Q40 25 16 15Z', bl, 0) + P('M26 4 L30 -1 L36 4Z', fn, 1.1) + [[46, 3], [50, 3], [54, 4], [58, 5]].map(([x, y]) => P('M' + x + ' ' + y + ' l2 -3 l2 3Z', fn, 0.8)).join('') + [[46, 22], [50, 21.6], [54, 21]].map(([x, y]) => P('M' + x + ' ' + y + ' l2 3 l2 -3Z', fn, 0.8)).join('') + P('M32 17 L28 24 L40 19Z', b, 1.1) + '<path d="M20 12 Q30 8 44 10" fill="none" stroke="' + mk + '" stroke-width="1" opacity="0.5"/>' + EY(63, 10, 1.9);
  else if (k === 'barracuda') o = P('M10 8 L-1 1 L4 8 L-1 15Z', fn) + P('M8 8 Q16 2 40 4 L66 6.4 L78 8 L66 9.8 L40 12 Q16 15 8 8Z', b, 1.4) + P('M20 9 Q42 14 66 9.8 L40 12.4Z', bl, 0) + [14, 20, 26, 32, 38, 44].map((x, i) => '<path d="M' + x + ' 4 l3 4 l-2 3" fill="none" stroke="' + mk + '" stroke-width="1.2" opacity="0.7"/>').join('') + P('M66 8 L78 8.4 L74 10.4 L66 10Z', bl, 1) + '<path d="M70 9 l1 1.4 M73 9.2 l0.6 1.2" stroke="#fff" stroke-width="0.7"/>' + P('M42 4 L46 0 L50 4.4Z', fn, 1) + P('M28 11.6 L26 15 L32 12.4Z', fn, 1) + EY(62, 6.4, 1.5);
  else if (k === 'reefshark') o = P('M14 16 L-1 3 L4 14 L10 16 L2 28 L8 18Z', b, 1.4) + P('M0 3 L4 14 L2 12Z', mk, 0) + P('M12 16 Q20 8 42 8 Q66 8 84 15 Q66 24 42 24 Q20 24 12 16Z', b, 1.5) + P('M18 17 Q44 24 80 16 Q44 26 18 17Z', bl, 0) + P('M36 9 L42 -1 L48 8Z', b, 1.3) + P('M40 2 L42 -1 L44 4Z', mk, 0) + P('M62 20 L66 25 L60 21Z', b, 1) + P('M44 22 L36 31 L52 24Z', b, 1.3) + P('M38 28 L36 31 L44 25Z', mk, 0) + P('M44 10 L38 1 L50 9Z', b, 1.3) + P('M40 4 L38 1 L45 6Z', mk, 0) + L('M76 13 q2 1 3 3', mk, 0.9) + EY(76, 12, 1.5) + '<path d="M70 10 v3 M72 10 v3 M74 10 v3" stroke="' + mk + '" stroke-width="0.7" opacity="0.6"/>';
  else if (k === 'whiteshark') o = P('M18 19 L-1 2 L6 17 L12 19 L2 34 L10 21Z', b, 1.5) + P('M16 19 Q24 8 52 8 Q82 8 104 18 Q82 29 52 29 Q24 29 16 19Z', b, 1.6) + P('M22 22 Q52 31 100 19 Q52 33 22 22Z', bl, 0) + P('M46 9 L54 -2 L63 9Z', b, 1.4) + P('M54 29 L44 40 L66 31Z', b, 1.3) + P('M52 11 L40 0 L62 10Z', b, 1.3) + P('M80 25 L85 31 L77 26Z', b, 1) + '<path d="M84 22 q4 1 9 -0.4" fill="none" stroke="' + INK + '" stroke-width="1.1"/>' + '<path d="M90 21 l1 2 M93 20.6 l0.6 2" stroke="#fff" stroke-width="0.8"/>' + EY(94, 16, 1.7) + '<path d="M80 14 q1 3 0 5 M83 14 q1 3 0 5 M86 14 q1 3 0 5" fill="none" stroke="' + mk + '" stroke-width="0.8" opacity="0.7"/>';
  else if (k === 'dolphin') o = P('M12 15 Q4 3 -1 5 Q4 11 8 14 Q4 19 -1 25 Q4 25 12 15Z', fn, 1.4) + P('M10 15 Q22 4 44 6 Q60 8 70 13 L80 14.4 L70 17.6 Q58 24 40 24 Q22 24 10 15Z', b, 1.5) + P('M20 17 Q42 25 68 17 Q44 27 20 17Z', bl, 0) + P('M34 7 L42 -1 L46 8Z', fn, 1.3) + P('M40 20 L34 29 L50 22Z', fn, 1.3) + P('M58 8 q4 -3 7 0', 'none', 1) + '<path d="M68 13 Q72 12 78 14" fill="none" stroke="' + INK + '" stroke-width="1"/>' + EY(66, 11.4, 1.5) + '<circle cx="52" cy="7.4" r="1" fill="' + INK + '"/>';
  else if (k === 'seal') o = P('M6 13 q-6 -7 -7 -2 q2 4 6 4Z M6 13 q-6 7 -7 2 q2 -4 6 -4Z', fn, 1.1) + E(24, 13, 19, 8.6, b) + [[14, 10], [20, 8], [26, 10], [32, 9], [18, 16], [26, 17], [34, 14], [22, 13]].map(([x, y]) => '<circle cx="' + x + '" cy="' + y + '" r="1.5" fill="' + mk + '" opacity="0.5"/>').join('') + P('M26 20 L22 26 L32 22Z', fn, 1.2) + P('M26 6 L22 0 L32 4Z', fn, 1.2) + E(42, 13, 7.4, 6, b) + E(48, 14, 2.6, 2, mk, 0.8) + '<path d="M46 16 l3 2 M44 17 l-1 2.4" stroke="' + bl + '" stroke-width="0.8"/>' + EY(44, 10, 1.5) + EY(44, 16, 1.5);
  else if (k === 'otter') o = P('M6 11 Q-1 8 -1 12 Q1 15 7 13Z', fn, 1.2) + E(22, 11, 15, 6.6, b) + E(24, 11, 8, 3.8, bl, 0) + P('M16 5 l-3 -4 l6 2Z M16 17 l-3 4 l6 -2Z', b, 1.1) + P('M30 4 L34 0 L36 5Z M30 18 L34 22 L36 17Z', b, 1.1) + E(38, 11, 7, 6, bl, 1.4) + E(43, 10, 2.4, 2, mk, 0.8) + P('M34 5 q0 -2 2 -2 M34 17 q0 2 2 2', b, 1) + EY(39, 8, 1.2) + EY(39, 14, 1.2) + '<path d="M42 13 l3 1 M42 14 l3 2" stroke="' + bl + '" stroke-width="0.7"/>';
  else if (k === 'whale') o = P('M30 36 Q18 20 -1 12 Q8 30 12 36 Q8 42 -1 60 Q18 52 30 36Z', fn, 1.8) + P('M26 36 Q40 18 80 18 Q130 18 160 32 Q150 42 130 46 Q80 52 40 48 Q26 44 26 36Z', b, 2) + P('M40 42 Q90 52 140 42 Q150 40 158 34 Q120 54 60 52Z', bl, 0) + [[96, 40], [100, 43], [104, 40], [108, 43], [112, 40]].map(([x, y]) => '<path d="M' + x + ' ' + y + ' h8" stroke="' + mk + '" stroke-width="0.9" opacity="0.5"/>').join('') + P('M82 20 L90 13 L98 20Z', b, 1.5) + P('M106 22 Q98 4 64 -1 Q78 12 94 28Z', fn, 1.7) + P('M106 50 Q98 68 64 73 Q78 60 94 44Z', fn, 1.7) + [[134, 27], [140, 28], [146, 29], [138, 32], [144, 33]].map(([x, y]) => '<circle cx="' + x + '" cy="' + y + '" r="1.4" fill="' + bl + '" opacity="0.7"/>').join('') + '<path d="M120 41 Q140 43 156 36" fill="none" stroke="' + INK + '" stroke-width="1.3"/>' + EY(148, 35, 1.6);
  else o = E(S.vw / 2, S.vh / 2, S.vw / 2 - 2, S.vh / 2 - 2, b);
  return o;
}

const MARP = {
  phyto:   { n: 'Phytoplankton', role: 'Producer: sunlit algae', zone: 'Surface water, to about 50 m', t: [-2, 34], eats: 'Sunlight, dissolved nutrients and carbon dioxide', note: 'Single-celled drifting algae. They make half the oxygen you breathe and feed everything in the sea. Blooms follow nutrient-rich upwelling and spring light, and colour the water green or brown.', c: ['#5aaa3c', '#d8c030', '#2a7a4a'] },
  zoo:     { n: 'Zooplankton', role: 'Drifting grazers', zone: 'Surface water; rises at night, sinks by day', t: [-2, 34], eats: 'Phytoplankton and other plankton', note: 'Copepods, krill and larvae of almost everything else. They graze algae and are eaten by sardines, jellyfish, mantas and whales. Many migrate hundreds of metres up and down each day.', c: ['#e8a868', '#f4d8b8', '#a85a38'] },
  kelp:    { n: 'Giant kelp', role: 'Producer: cold-water forest', zone: 'Cold rocky shallows, 3 to 16 tiles out', t: [4, 20], eats: 'Sunlight and cold, nutrient-rich water', note: 'Brown seaweed that grows half a metre a day, anchored to rock by a holdfast and held up by gas bladders. Forms underwater forests sheltering fish, otters and urchins. Dies back in warm water.', c: ['#7e8a2a', '#c8b050', '#5a5a1e'] },
  seagrass:{ n: 'Seagrass meadow', role: 'Producer: flowering sea plant', zone: 'Sandy shallows, to about 8 tiles out', t: [4, 34], eats: 'Sunlight; roots in sand', note: 'Not a seaweed but a true flowering plant. Meadows hold sand in place, shelter young fish and feed turtles and dugongs. Slow to regrow once grazed or dredged.', c: ['#46a85e', '#2f8a4c', '#7acb7c'] },
  coral:   { n: 'Reef coral', role: 'Reef builder', zone: 'Clear, warm shallows (mean temperature over 18 C)', t: [18, 32], eats: 'Algae living in its tissue, plus plankton caught by tentacles', note: 'Colonies of tiny polyps that lay down limestone, growing a reef a few millimetres a year. They bleach and die if the water gets too hot and are grazed by parrotfish and urchins.', c: ['#e0708c', '#f0a860', '#a860c0'] },
  seaweed: { n: 'Rockweed and dulse', role: 'Intertidal seaweed', zone: 'The tideline: rocks and beach edge', t: [-2, 28], eats: 'Sunlight; survives drying between tides', note: 'Leathery brown and red weeds with air bladders that cling to tideline rocks. Crabs shelter in it, shore foragers gather it for food and fertiliser, and it washes up as wrack that feeds beach insects.', c: ['#7a6a28', '#a8483a', '#4a5a1e'] },
  anemone: { n: 'Sea anemone', role: 'Reef predator', zone: 'Coral reef and rock pools', t: [10, 32], eats: 'Plankton and small fish that brush its stinging tentacles', note: 'A soft flower-shaped animal with stinging tentacles. Clownfish live unharmed in it and are fed and protected in return.', c: ['#e85a8a', '#f4b8d0', '#8a2a4a'] }
};

function marpSvg(k) {
  const S = MARP[k], [a, b, c] = S.c, E = (cx, cy, rx, ry, f, w) => '<ellipse cx="' + cx + '" cy="' + cy + '" rx="' + rx + '" ry="' + ry + '" fill="' + f + '" stroke="' + INK + '" stroke-width="' + (w === undefined ? 1.2 : w) + '"/>', P = (d, f, w) => '<path d="' + d + '" fill="' + f + '" stroke="' + INK + '" stroke-width="' + (w === undefined ? 1.2 : w) + '" stroke-linejoin="round" stroke-linecap="round"/>', L = (d, col, w) => '<path d="' + d + '" fill="none" stroke="' + col + '" stroke-width="' + w + '" stroke-linecap="round"/>';
  let o = '';
  if (k === 'phyto') { const diatom = (x, y, r, col) => { let q = ''; for (let i = 0; i < 12; i++) { const an = i / 12 * Math.PI * 2; q += '<path d="M' + (x + Math.cos(an) * r).toFixed(1) + ' ' + (y + Math.sin(an) * r).toFixed(1) + ' L' + (x + Math.cos(an) * (r + 3)).toFixed(1) + ' ' + (y + Math.sin(an) * (r + 3)).toFixed(1) + '" stroke="' + col + '" stroke-width="0.8"/>'; } return q + E(x, y, r, r, col, 1) + E(x, y, r * 0.55, r * 0.55, '#e8f4c8', 0.6); }; o = diatom(12, 12, 5, a) + diatom(30, 10, 4, b) + diatom(22, 26, 6, a) + P('M6 30 l3 -6 l3 6 l-3 3Z', c, 1) + L('M34 24 q3 2 0 5 q-3 2 0 5', b, 2.4) + E(10, 32, 3, 3, b, 1); }
  else if (k === 'zoo') o = E(14, 14, 7, 4, a, 1.2) + L('M7 14 L1 8 M7 14 L1 20 M20 12 L30 6 M20 16 L30 22', INK, 2.4) + L('M7 14 L1 8 M7 14 L1 20 M20 12 L30 6 M20 16 L30 22', b, 1) + E(30, 30, 2, 5, c, 1) + E(26, 28, 5, 2.4, a, 1) + L('M22 28 L18 24 M22 28 L18 32', b, 1) + '<circle cx="30" cy="30" r="1" fill="#101010"/>' + P('M8 28 l4 -4 l4 4 l-4 6Z', b, 1) + '<circle cx="12" cy="29" r="1" fill="' + c + '"/>';
  else if (k === 'kelp') o = [10, 20, 30].map((x, i) => L('M' + x + ' 38 Q' + (x + (i % 2 ? 4 : -4)) + ' 24 ' + x + ' 6', INK, 4) + L('M' + x + ' 38 Q' + (x + (i % 2 ? 4 : -4)) + ' 24 ' + x + ' 6', c, 2.2) + [10, 17, 24, 31].map((y, j) => P('M' + x + ' ' + y + ' q' + (j % 2 ? 7 : -7) + ' -3 ' + (j % 2 ? 9 : -9) + ' -8 q-' + (j % 2 ? 6 : -6) + ' 1 -' + (j % 2 ? 9 : -9) + ' 8Z', j % 2 ? a : b, 1)).join('') + E(x, 5, 2.2, 2.6, b, 1)).join('') + P('M4 38 h32 l-3 2 h-26Z', '#6a6a70', 1.2);
  else if (k === 'seagrass') o = [[6, 6], [11, 10], [16, 4], [21, 9], [26, 5], [31, 11], [14, 14], [24, 15]].map(([x, h], i) => L('M' + x + ' 38 Q' + (x + (i % 2 ? 3 : -3)) + ' ' + (38 - h * 1.5) + ' ' + (x + (i % 2 ? 2 : -2)) + ' ' + h, INK, 3.2) + L('M' + x + ' 38 Q' + (x + (i % 2 ? 3 : -3)) + ' ' + (38 - h * 1.5) + ' ' + (x + (i % 2 ? 2 : -2)) + ' ' + h, i % 2 ? b : a, 1.6)).join('') + '<path d="M2 38 h36" stroke="#d8c898" stroke-width="3"/>' + '<path d="M2 38 h36" stroke="' + INK + '" stroke-width="0.8"/>';
  else if (k === 'coral') o = E(11, 30, 9, 7, a, 1.3) + '<path d="M5 28 Q11 25 17 29 M6 32 Q11 29 16 33" stroke="#8a3048" stroke-width="1" fill="none"/>' + [0, 1, 2, 3, 4].map((i) => P('M' + (22 + i * 3.6) + ' 36 L' + (22 + i * 3.6) + ' ' + (20 - (i % 2) * 5) + ' l-1.5 -3 M' + (22 + i * 3.6) + ' ' + (28 - (i % 2) * 2) + ' l3 -4', b, 0) + L('M' + (22 + i * 3.6) + ' 36 L' + (22 + i * 3.6) + ' ' + (20 - (i % 2) * 5), INK, 3.4) + L('M' + (22 + i * 3.6) + ' 36 L' + (22 + i * 3.6) + ' ' + (20 - (i % 2) * 5), b, 1.8)).join('') + P('M4 22 Q2 8 14 4 Q26 2 24 14 Q22 20 14 22Z', c, 1.3) + '<path d="M8 20 Q8 10 14 7 M13 21 Q13 12 18 8" stroke="#fff" stroke-width="0.9" fill="none" opacity="0.8"/>' + '<path d="M2 38 h36" stroke="#d8c898" stroke-width="3"/>';
  else if (k === 'seaweed') o = P('M4 38 h32 l-4 -3 h-24Z', '#6a6a70', 1.2) + [10, 19, 28].map((x, i) => L('M' + x + ' 35 Q' + (x + (i - 1) * 4) + ' 22 ' + (x + (i - 1) * 7) + ' 8', INK, 4) + L('M' + x + ' 35 Q' + (x + (i - 1) * 4) + ' 22 ' + (x + (i - 1) * 7) + ' 8', i === 1 ? b : a, 2.2) + [14, 22, 29].map((y, j) => E(x + (i - 1) * (30 - y) / 6 + (j % 2 ? 2 : -2), y, 2.2, 2.6, c, 0.9)).join('') + P('M' + (x + (i - 1) * 7) + ' 8 q-4 -2 -3 -6 q4 1 3 6 q4 -5 6 -4 q-1 4 -6 5Z', i === 1 ? b : a, 0.9)).join('');
  else if (k === 'anemone') { let t = ''; for (let i = 0; i < 16; i++) { const an = i / 16 * Math.PI * 2, x1 = 20 + Math.cos(an) * 6, y1 = 22 + Math.sin(an) * 5, x2 = 20 + Math.cos(an + 0.3) * 16, y2 = 22 + Math.sin(an + 0.3) * 14; t += L('M' + x1.toFixed(1) + ' ' + y1.toFixed(1) + ' Q' + (20 + Math.cos(an) * 12).toFixed(1) + ' ' + (22 + Math.sin(an) * 11).toFixed(1) + ' ' + x2.toFixed(1) + ' ' + y2.toFixed(1), INK, 3.6) + L('M' + x1.toFixed(1) + ' ' + y1.toFixed(1) + ' Q' + (20 + Math.cos(an) * 12).toFixed(1) + ' ' + (22 + Math.sin(an) * 11).toFixed(1) + ' ' + x2.toFixed(1) + ' ' + y2.toFixed(1), i % 2 ? a : b, 2) + E(x2, y2, 1.6, 1.6, c, 0.8); } o = t + E(20, 22, 6, 5, a, 1.3) + E(20, 22, 2.6, 2, c, 0.8); }
  return o;
}

function fishSvg(k) {
  const S = SEA[k], [b, bl, fn, mk] = S.c, P = (d, f, w) => '<path d="' + d + '" fill="' + f + '" stroke="' + INK + '" stroke-width="' + (w || 1.5) + '" stroke-linejoin="round" stroke-linecap="round"/>', E = (cx, cy, rx, ry, f, w) => '<ellipse cx="' + cx + '" cy="' + cy + '" rx="' + rx + '" ry="' + ry + '" fill="' + f + '" stroke="' + INK + '" stroke-width="' + (w == null ? 1.5 : w) + '"/>', EY = (x, y) => '<circle cx="' + x + '" cy="' + y + '" r="1.3" fill="#f4efe0" stroke="' + INK + '" stroke-width="0.8"/><circle cx="' + (x + 0.3) + '" cy="' + y + '" r="0.55" fill="' + INK + '"/>';
  let o = '';
  if (S.plan === 'minnow') o = P('M3 10 L-1 5 L-1 15Z', fn) + E(15, 10, 14, 4.6, b) + P('M5 10 Q16 12.6 27 10 Q16 14.6 5 10Z', bl, 0) + '<path d="M3 10 H26" stroke="' + mk + '" stroke-width="1.2" opacity="0.7"/>' + EY(25, 8.8) + P('M14 14 L11 18 L17 15Z', fn, 1.1);
  else if (S.plan === 'trout') o = P('M4 10 L-1 4 L0 10 L-1 16Z', fn) + E(18, 10, 17, 5.4, b) + P('M7 10 Q19 13.4 33 10 Q19 15.4 7 10Z', bl, 0) + [[12, 8], [17, 7], [22, 8.4], [15, 11.4], [21, 11.6], [26, 9.6]].map(([x, y]) => '<circle cx="' + x + '" cy="' + y + '" r="0.9" fill="' + mk + '"/>').join('') + P('M28 8.6 Q32 8 33 10', b, 0) + EY(31, 8.6) + P('M13 14 L9 19 L18 15.6Z', fn, 1.1) + P('M13 6 L9 1 L18 4.4Z', fn, 1.1);
  else if (S.plan === 'carp') o = P('M4 10 L-1 3 L1 10 L-1 17Z', fn) + E(19, 10, 18, 7.4, b) + P('M6 10 Q20 15.6 35 10 Q20 17.4 6 10Z', bl, 0) + [8, 12, 16, 20, 24, 28].map((x, i) => '<path d="M' + x + ' 6 q3 3 0 7" fill="none" stroke="' + mk + '" stroke-width="0.8" opacity="0.6"/>').join('') + EY(33, 7.6) + '<path d="M37 11 q2 2 4 1 M37 9 q2 -2 4 -1" stroke="' + INK + '" stroke-width="1" fill="none"/>' + P('M14 16 L9 20 L20 17.4Z', fn, 1.1) + P('M14 4 L9 0 L20 2.6Z', fn, 1.1);
  else if (S.plan === 'pike') o = P('M5 10 L-1 3 L1 10 L-1 17Z', fn) + P('M3 10 Q14 4 30 7 L40 9.4 L40 11.4 L30 13 Q14 16 3 10Z', b) + P('M8 10 Q20 13.6 36 10.6 L30 13 Q14 16 8 10Z', bl, 0) + [[9, 8], [14, 7.4], [19, 8], [24, 8.4]].map(([x, y]) => '<rect x="' + x + '" y="' + y + '" width="3" height="1.6" rx="0.8" fill="' + mk + '" opacity="0.65"/>').join('') + EY(32, 8.4) + P('M9 12 L5 17 L13 13.6Z', fn, 1.1) + P('M7 8 L3 3 L12 7Z', fn, 1.1);
  else if (S.plan === 'catfish') o = P('M5 10 L0 4 L1 10 L0 16Z', fn) + P('M4 10 Q10 3 24 3.6 Q35 4.6 36 10 Q35 15.4 24 16.4 Q10 17 4 10Z', b) + P('M8 10 Q20 15 34 10 Q30 15.6 24 16.4 Q10 17 4 10Z', bl, 0) + EY(31, 6.4) + EY(31, 13.6) + '<path d="M35 7 Q39 4 41 6 M35 13 Q39 16 41 14 M36 9 Q39 9 41 8 M36 11 Q39 11 41 12" stroke="' + INK + '" stroke-width="1.1" fill="none" stroke-linecap="round"/>' + P('M17 3.6 L14 0 L22 3Z', fn, 1.1);
  else if (S.plan === 'eel') o = '<path d="M0 12 C6 4 10 16 17 10 S28 5 34 9 L38 9.4 L38 11 L34 11.4 C28 8 24 16 17 13 S6 7 0 12Z" fill="' + b + '" stroke="' + INK + '" stroke-width="1.6" stroke-linejoin="round"/><path d="M6 12 C9 8 12 14 17 11.6 S27 9 32 10.6" fill="none" stroke="' + bl + '" stroke-width="1.2" opacity="0.8"/>' + EY(34, 9.4) + P('M13 8 L11 5.6 L16 8Z', fn, 1);
  else if (S.plan === 'crayfish') o = '<path d="M2 6 Q-1 10 2 14 M5 4 L3 2 M5 16 L3 18" stroke="' + INK + '" stroke-width="1.4" fill="none" stroke-linecap="round"/>' + '<path d="M8 10 L3 7 M8 10 L3 13 M10 7 L6 3 M10 13 L6 17 M13 6 L10 2 M13 14 L10 18" stroke="' + INK + '" stroke-width="1.6" stroke-linecap="round"/><path d="M8 10 L3 7 M8 10 L3 13 M10 7 L6 3 M10 13 L6 17 M13 6 L10 2 M13 14 L10 18" stroke="' + fn + '" stroke-width="0.7" stroke-linecap="round"/>' + P('M2 10 L9 7 L9 13Z', fn) + E(15, 10, 8, 4.2, b) + P('M20 8.6 Q27 4 36 3 Q39 6 33 9 Q27 9 22 10Z', b, 1.3) + P('M20 11.4 Q27 16 36 17 Q39 14 33 11 Q27 11 22 10Z', b, 1.3) + E(21, 8, 1.2, 1.2, '#f4efe0', 0.6) + E(21, 12, 1.2, 1.2, '#f4efe0', 0.6);
  else if (S.plan === 'turtle') o = P('M18 3 L22 0 L24 4Z', fn, 1.2) + P('M14 15 L10 19 L17 17Z', fn, 1.2) + P('M14 5 L10 1 L17 3Z', fn, 1.2) + P('M30 15 L32 19 L26 17Z', fn, 1.2) + P('M30 5 L32 1 L26 3Z', fn, 1.2) + P('M4 10 L0 9 L1 11Z', fn, 1.2) + E(22, 10, 11, 8, b, 1.7) + '<path d="M22 2.6 V17.4 M14 10 H30 M16 5 L28 15 M28 5 L16 15" stroke="' + mk + '" stroke-width="0.9" opacity="0.7"/>' + E(22, 10, 5, 3.8, fn, 1) + P('M32 8 Q38 8.4 38 10 Q38 11.6 32 12Z', bl, 1.3) + EY(36, 9);
  else if (S.plan === 'perch') o = P('M4 10 L-1 4 L0 10 L-1 16Z', fn) + E(18, 10.4, 16, 6.4, b) + P('M7 11 Q19 16 31 11 Q19 17 7 11Z', bl, 0) + [10, 14, 18, 22, 26].map((x) => '<path d="M' + x + ' 5 q1.6 5.4 0 11" fill="none" stroke="' + mk + '" stroke-width="1.7" opacity="0.8"/>').join('') + P('M10 5 L12 0 L15 3 L17 0 L21 5Z', fn, 1.1) + P('M22 5 Q26 1 28 5Z', b, 1) + P('M14 15 L11 19.4 L19 16Z', fn, 1.1) + P('M6 12 L3 16 L10 14Z', fn, 1.1) + EY(30, 8.6);
  else if (S.plan === 'salmon') o = P('M3 10 L-2 3 L0 10 L-2 17Z', fn) + P('M3 10 Q12 3.6 26 5 Q34 6 37 9.6 Q34 12.4 26 14.6 Q12 17 3 10Z', b) + P('M6 11.4 Q20 16.4 35 11 Q28 15.2 20 15.8 Q10 16.4 6 11.4Z', bl, 0) + '<path d="M6 10 Q20 6.4 34 9" stroke="' + fn + '" stroke-width="2" fill="none" opacity="0.55"/>' + [[12, 7.6], [17, 6.8], [22, 7.2], [27, 7.4], [15, 9], [24, 9.4]].map(([x, y]) => '<circle cx="' + x + '" cy="' + y + '" r="0.8" fill="' + mk + '"/>').join('') + P('M35 10.6 Q38 12.6 39.4 11.4', b, 1) + EY(32.4, 8.4) + P('M15 5.6 L12 1.6 L20 4.8Z', fn, 1.1) + P('M12 14.6 L9 18.6 L18 16Z', fn, 1.1) + P('M8 6.2 L6 4.2 L10 5.4Z', b, 0.8);
  else if (S.plan === 'sturgeon') o = P('M6 10 L-2 1.6 L1.6 9.6 L-1 14 L2 12Z', fn) + P('M5 10 Q12 4.4 30 6.4 L44 9.6 L30 13.6 Q14 15.6 5 10Z', b) + P('M8 11.6 Q22 15.2 42 10.2 L30 13.6 Q14 15.6 8 11.6Z', bl, 0) + [10, 15, 20, 25, 30].map((x) => P('M' + x + ' 6.4 l1.8 -2.8 l1.8 2.8Z', mk, 0.9)).join('') + [11, 17, 23, 29].map((x) => P('M' + x + ' 9.4 l1 -1.3 l1 1.3 l-1 1.3Z', bl, 0.8)).join('') + EY(32, 8.2) + '<path d="M40 11.6 l-0.6 3 M42 11 l-0.2 3.4" stroke="' + INK + '" stroke-width="1" stroke-linecap="round"/>' + P('M15 13.6 L12 17.6 L21 14.6Z', fn, 1.1);
  else if (S.plan === 'frog') o = P('M2 14 Q-1 11 1 8 Q5 11 10 13Z', b, 1.3) + P('M10 15 Q4 19 0 17.6 Q6 15 10 12Z', b, 1.3) + '<path d="M0 18 L-1 20 M2 18.4 L2 20.4 M4 18 L5 20" stroke="' + INK + '" stroke-width="1" stroke-linecap="round"/>' + E(17, 10.6, 11, 6.6, b, 1.7) + P('M9 13 Q18 17.6 27 13 Q18 16 9 13Z', bl, 0) + [[13, 8.6], [18, 7.6], [22, 9.4]].map(([x, y]) => '<circle cx="' + x + '" cy="' + y + '" r="1.2" fill="' + mk + '" opacity="0.75"/>').join('') + E(27, 8.4, 5, 4.6, b, 1.6) + '<path d="M28 11.4 Q33 12.4 37 11" stroke="' + INK + '" stroke-width="1.4" fill="none" stroke-linecap="round"/>' + '<circle cx="28.4" cy="4.8" r="2.7" fill="' + b + '" stroke="' + INK + '" stroke-width="1.4"/><circle cx="28.8" cy="4.6" r="1.5" fill="#e8c840" stroke="' + INK + '" stroke-width="0.7"/><circle cx="29.2" cy="4.6" r="0.7" fill="' + INK + '"/>' + '<path d="M18 14 Q22 19 29 18 L33 19.6 M22 15 L25 19" stroke="' + INK + '" stroke-width="2.2" fill="none" stroke-linecap="round"/><path d="M18 14 Q22 19 29 18 L33 19.6" stroke="' + b + '" stroke-width="0.9" fill="none" stroke-linecap="round"/>';
  else if (S.plan === 'newt') o = '<path d="M0 8 Q5 15 12 11 Q20 8 28 10 Q33 11 36 9.4" fill="none" stroke="' + INK + '" stroke-width="6.4" stroke-linecap="round"/><path d="M0 8 Q5 15 12 11 Q20 8 28 10 Q33 11 36 9.4" fill="none" stroke="' + b + '" stroke-width="4.4" stroke-linecap="round"/><path d="M3 11.4 Q8 15.6 13 12.6 Q20 10.6 28 12" fill="none" stroke="' + bl + '" stroke-width="1.8" stroke-linecap="round"/><path d="M4 5.4 Q8 9 12 7.2 Q20 4.8 27 6.2" fill="none" stroke="' + fn + '" stroke-width="1.4" opacity="0.8"/>' + [[12, 11.4], [18, 11.2], [24, 11.8]].map(([x, y]) => '<circle cx="' + x + '" cy="' + y + '" r="0.7" fill="' + mk + '"/>').join('') + '<path d="M26 12.6 l3 4 M16 12 l-1 4.6 M31 11 l4 3.4 M12 12 l-3 3.6" stroke="' + INK + '" stroke-width="1.5" stroke-linecap="round"/>' + P('M33 7 Q38 6.6 40 9 Q38 12 33 11.4Z', b, 1.3) + EY(36.6, 8.4);
  else if (S.plan === 'mussel') o = '<path d="M3 16 H39" stroke="#6a5a3a" stroke-width="3" stroke-linecap="round"/>' + P('M4 15 Q4 4 18 3 Q34 3 38 13 Q38 16 33 16 H8Q4 16 4 15Z', b, 1.7) + '<path d="M7 14 Q9 7 18 6 Q28 6 33 12 M10 14.4 Q12 9.6 19 8.8 Q26 9 30 13" fill="none" stroke="' + bl + '" stroke-width="0.9" opacity="0.7"/>' + '<path d="M8 15 Q20 12.4 36 14.6" fill="none" stroke="' + mk + '" stroke-width="1.5" opacity="0.8"/>' + P('M30 14.4 Q35 12 38 14 Q36 16.6 30 16Z', '#e8b0a0', 1) + '<path d="M16 4 Q17 1 18 3" fill="none" stroke="' + INK + '" stroke-width="1"/>';
  return '<g transform="' + (S.plan === 'sturgeon' ? 'translate(1.6 0.6) scale(0.9)' : S.plan === 'salmon' ? 'translate(1 0)' : 'translate(0.4 0)') + '">' + o + '</g>';
}

function darker(hex, f) { const n = parseInt(hex.slice(1), 16); return '#' + [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => Math.round(v * f).toString(16).padStart(2, '0')).join(''); }

const svgBg = (w, h, inner) => "url('data:image/svg+xml;utf8," + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="' + w + '" height="' + h + '" viewBox="0 0 ' + w + ' ' + h + '">' + inner + '</svg>').replace(/'/g, '%27') + "')";

const CI = (cx, cy, r, f, ex) => '<circle cx="' + cx.toFixed(1) + '" cy="' + cy.toFixed(1) + '" r="' + r.toFixed(1) + '" fill="' + f + '"' + (ex || '') + '/>';

function lobesOf(v, n, R, sp, cx0, cy0) {
  const out = [[cx0, cy0, R]];
  for (let i = 0; i < n; i++) { const a = v * 0.9 + i / n * Math.PI * 2; out.push([cx0 + Math.cos(a) * sp * (0.92 + 0.12 * ((i + v) % 2)), cy0 + Math.sin(a) * sp * (0.92 + 0.1 * ((i * 3 + v) % 3) / 2), R * (0.56 + 0.12 * ((i + v) % 3))]); }
  return out;
}

const poly = (pts, f, ex) => '<polygon points="' + pts + '" fill="' + f + '"' + (ex || '') + '/>';

function bushSvg(f) {
  const lb = [[19, 18, 10], [10.5, 20.5, 7.5], [27.5, 20.5, 7.5], [14.5, 11.5, 7], [24, 11, 7]];
  let o = '<ellipse cx="19" cy="31" rx="14" ry="2.6" fill="' + INK + '" opacity="0.3"/><path d="M13 29 L11 33 M19 30 L19 34 M25 29 L27 33" stroke="' + INK + '" stroke-width="2.6" stroke-linecap="round"/><path d="M13 29 L11 33 M19 30 L19 34 M25 29 L27 33" stroke="#6a4a30" stroke-width="1.2" stroke-linecap="round"/>' + canopy3(lb, f ? '#4b7a3c' : '#6b7a4a', 19, 17);
  const spots = [[12, 12], [24, 10], [18, 19], [9, 21], [28, 22], [16, 25], [22, 24], [19, 8], [30, 15]];
  if (!f) o += '<path d="M10 22 L16 14 M24 10 L28 18 M18 24 L19 16" stroke="' + INK + '" stroke-width="1.2" fill="none" opacity="0.6"/>';
  spots.slice(0, f * 3).forEach(([x, y]) => { o += CI(x, y, 2.5, INK) + CI(x, y, 1.7, '#d9503c') + CI(x - 0.6, y - 0.6, 0.55, '#f4b0a0'); });
  return o;
}

function forageSvg(k, f) {
  const sh = '<ellipse cx="19" cy="31" rx="14" ry="2.6" fill="' + INK + '" opacity="0.3"/>';
  const L = (d, c, w) => '<path d="' + d + '" stroke="' + INK + '" stroke-width="' + (w + 1.6) + '" fill="none" stroke-linecap="round"/><path d="' + d + '" stroke="' + c + '" stroke-width="' + w + '" fill="none" stroke-linecap="round"/>';
  const E = (cx, cy, rx, ry, c) => '<ellipse cx="' + cx + '" cy="' + cy + '" rx="' + (rx + 1) + '" ry="' + (ry + 1) + '" fill="' + INK + '"/><ellipse cx="' + cx + '" cy="' + cy + '" rx="' + rx + '" ry="' + ry + '" fill="' + c + '"/>';
  let o = sh;
  if (k === 'tuber') {
    const g = f ? '#5f9040' : '#8a7a4a', g2 = f ? '#86b85a' : '#a09060', lf = (x, y, rot, len, c) => '<ellipse cx="' + x + '" cy="' + y + '" rx="' + len + '" ry="' + (len * 0.48).toFixed(1) + '" transform="rotate(' + rot + ' ' + x + ' ' + y + ')" fill="' + c + '" stroke="' + INK + '" stroke-width="1.1"/>';
    o += E(19, 30, 14, 3.6, '#6a4a30') + '<path d="M8 29 q3 -2 6 -1 M22 29 q4 -2 8 0" stroke="#8a6a46" stroke-width="1" fill="none"/>';
    [[9, 12, -30], [16, 7, 10], [23, 9, -10], [30, 14, 30]].forEach(([x, y, r], i) => { o += L('M' + (x + (i % 2 ? 1 : -1) * 1.5) + ' 29 Q' + x + ' 20 ' + x + ' ' + (y + 4), g, 1.8) + lf(x - 3, y + 3, -35 + r, 4.4, g2) + lf(x + 3, y + 3, 35 + r, 4.4, g) + lf(x, y, r, 4, g2); });
    [[11, 30, '#c9a46a'], [20, 31, '#a56a9a'], [28, 30, '#c9a46a']].slice(0, f).forEach(([x, y, c]) => { o += '<path d="M' + (x - 1) + ' ' + (y + 2) + ' l-2 3 M' + (x + 1) + ' ' + (y + 2) + ' l2 3" stroke="' + INK + '" stroke-width="1" fill="none"/>' + E(x, y, 4.6, 3.2, c) + '<path d="M' + (x - 2.4) + ' ' + (y - 1) + ' q2.4 -1.8 4.8 0" stroke="#fff7" stroke-width="0.9" fill="none"/>'; });
    if (!f) o += '<path d="M8 31 h6 M22 31 h6" stroke="#6a4a30" stroke-width="1.4"/>';
  } else if (k === 'nut') {
    const lobe = (x, y, r, c) => '<path d="M' + x + ' ' + y + ' q-4 -3 -2 -6 q2 0 3 -3 q2 3 4 3 q2 3 -1 6 q-1 2 -4 0Z" transform="rotate(' + r + ' ' + x + ' ' + y + ')" fill="' + c + '" stroke="' + INK + '" stroke-width="1.1" stroke-linejoin="round"/>';
    o += E(19, 28.5, 16, 4.6, '#6e4a26') + lobe(7, 28, -30, '#a8742e') + lobe(29, 27, 40, '#8a5a24') + lobe(18, 25, 10, '#b88638') + '<path d="M5 29 l3 -2 M26 26 l5 2 M15 31 l5 1" stroke="#4a3018" stroke-width="1" fill="none"/>';
    [[10, 26], [17, 28], [25, 26], [30, 29], [14, 31], [22, 31]].slice(0, f * 2).forEach(([x, y]) => { o += E(x, y + 1, 2.9, 3.5, '#8a5524') + '<path d="M' + (x - 1.2) + ' ' + y + ' q0.6 2 0.4 4" stroke="#c8944a" stroke-width="0.9" fill="none"/><path d="M' + (x - 3.2) + ' ' + (y - 0.6) + ' a3.2 2.6 0 0 1 6.4 0Z" fill="#5a3a1a" stroke="' + INK + '" stroke-width="1" stroke-linejoin="round"/><path d="M' + (x - 2) + ' ' + (y - 1.6) + ' h4 M' + (x - 1.4) + ' ' + (y - 0.2) + ' h2.8" stroke="#8a6a3a" stroke-width="0.7"/>'; });
    if (f >= 2) o += E(32, 24, 3.4, 3.4, '#6a8a3a') + '<path d="M30 22 l-1.4 -1.8 M32 21 v-2.2 M34 22 l1.4 -1.8" stroke="' + INK + '" stroke-width="1"/>';
  } else if (k === 'mushroom') {
    const cs = [[10, 26, 7, '#c2503c', 1], [22, 22, 8, '#8a5a3a', 0], [30, 28, 5.5, '#e8dcc4', 0], [17, 29, 5, '#b8683a', 0]];
    o += E(19, 30, 13, 2.4, '#4a6a34');
    if (!f) o += CI(14, 29, 1.6, '#9a8a6a') + CI(24, 29, 1.3, '#9a8a6a');
    cs.slice(0, f + 1).forEach(([x, y, rr, c, dots]) => { o += '<rect x="' + (x - 1.6) + '" y="' + y + '" width="3.2" height="' + (31 - y) + '" rx="1.2" fill="#efe6d2" stroke="' + INK + '" stroke-width="1"/><path d="M' + (x - rr) + ' ' + (y + 1) + ' a' + rr + ' ' + (rr * 0.8).toFixed(1) + ' 0 0 1 ' + (rr * 2) + ' 0Z" fill="' + c + '" stroke="' + INK + '" stroke-width="1.2"/>'; if (dots) o += CI(x - 2.5, y - 2, 1, '#fff') + CI(x + 1.5, y - 3, 1, '#fff') + CI(x + 3, y, 0.8, '#fff'); });
  } else if (k === 'nightshade') {
    const lb = [[19, 18, 10], [10.5, 20.5, 7.5], [27.5, 20.5, 7.5], [14.5, 11.5, 7], [24, 11, 7]];
    o += '<path d="M13 29 L11 33 M19 30 L19 34 M25 29 L27 33" stroke="' + INK + '" stroke-width="2.6" stroke-linecap="round"/><path d="M13 29 L11 33 M19 30 L19 34 M25 29 L27 33" stroke="#4a3a4a" stroke-width="1.2" stroke-linecap="round"/>' + canopy3(lb, f ? '#3f5a3c' : '#6b6a4a', 19, 17);
    [[12, 12], [24, 10], [18, 19], [9, 21], [28, 22], [16, 25], [22, 24], [19, 8], [30, 15]].slice(0, f * 3).forEach(([x, y]) => { o += CI(x, y, 2.4, INK) + CI(x, y, 1.6, '#3a1f4a') + CI(x - 0.6, y - 0.6, 0.55, '#b89ad0'); });
    if (f) o += '<path d="M15 14 l-1.5 -2.5 M26 17 l2 -2" stroke="#9a6ad0" stroke-width="1.2"/>';
  } else if (k === 'toadstool') {
    o += E(19, 30, 13, 2.4, '#4a5a44');
    [[11, 25, 6.5], [23, 21, 7.5], [29, 28, 5]].slice(0, f + 1).forEach(([x, y, rr]) => { o += '<rect x="' + (x - 1.7) + '" y="' + y + '" width="3.4" height="' + (31 - y) + '" rx="1.2" fill="#f2efe4" stroke="' + INK + '" stroke-width="1"/><path d="M' + (x - 2.6) + ' ' + (y + 2.6) + ' q2.6 2 5.2 0" stroke="' + INK + '" stroke-width="1" fill="#e8e4d4"/><path d="M' + (x - rr) + ' ' + (y + 1) + ' a' + rr + ' ' + (rr * 0.8).toFixed(1) + ' 0 0 1 ' + (rr * 2) + ' 0Z" fill="#cfd8a0" stroke="' + INK + '" stroke-width="1.2"/>' + CI(x - rr * 0.4, y - 1.5, 0.9, '#fff') + CI(x + rr * 0.3, y - 2.6, 0.8, '#fff'); });
  } else if (k === 'herb') {
    o += E(19, 30, 12, 2.2, '#4f7a3a');
    [[10, 12], [17, 6], [24, 10], [30, 15], [14, 18]].forEach(([x, y], i) => { o += L('M' + (x + (i % 2 ? 1 : -1)) + ' 31 Q' + x + ' 22 ' + x + ' ' + y, '#6fa04a', 1.6); o += '<ellipse cx="' + (x - 2.4) + '" cy="' + (y + 6) + '" rx="3" ry="1.3" transform="rotate(-30 ' + (x - 2.4) + ' ' + (y + 6) + ')" fill="#7fb858" stroke="' + INK + '" stroke-width="0.8"/><ellipse cx="' + (x + 2.4) + '" cy="' + (y + 9) + '" rx="3" ry="1.3" transform="rotate(30 ' + (x + 2.4) + ' ' + (y + 9) + ')" fill="#7fb858" stroke="' + INK + '" stroke-width="0.8"/>'; if (f && i < 1 + f) o += CI(x, y, 2.3, INK) + CI(x, y, 1.6, '#f0d84a') + CI(x, y, 0.6, '#d89a2a'); });
  } else if (k === 'grain') {
    const x0 = [7, 11, 15, 19, 23, 27, 31];
    x0.forEach((x, i) => { const h = f ? 22 - (i % 3) * 3 : 8 - (i % 2) * 2; o += L('M' + x + ' 31 Q' + (x + (i % 2 ? 2 : -2)) + ' ' + (31 - h / 2) + ' ' + (x + (i % 2 ? 3 : -3)) + ' ' + (31 - h), f ? '#b8b25a' : '#8a7a4a', 1.3); if (f && i < 2 + f * 2) { const hx = x + (i % 2 ? 3 : -3), hy = 31 - h; o += E(hx, hy - 1, 1.9, 4, '#e0b84c') + '<path d="M' + hx + ' ' + (hy - 5) + ' v-2" stroke="' + INK + '" stroke-width="0.8"/>'; } });
  } else if (k === 'cactus') {
    const pad = (x, y, rx, ry, r, c) => '<ellipse cx="' + x + '" cy="' + y + '" rx="' + (rx + 1) + '" ry="' + (ry + 1) + '" transform="rotate(' + r + ' ' + x + ' ' + y + ')" fill="' + INK + '"/><ellipse cx="' + x + '" cy="' + y + '" rx="' + rx + '" ry="' + ry + '" transform="rotate(' + r + ' ' + x + ' ' + y + ')" fill="' + c + '"/><ellipse cx="' + (x - rx * 0.25) + '" cy="' + (y - ry * 0.3) + '" rx="' + rx * 0.55 + '" ry="' + ry * 0.45 + '" transform="rotate(' + r + ' ' + x + ' ' + y + ')" fill="#8bc070" opacity="0.5"/>';
    const sp = (pts) => pts.map(([x, y]) => '<path d="M' + x + ' ' + y + ' l-1 -1.6 M' + x + ' ' + y + ' l1 -1.6" stroke="#efe8cf" stroke-width="0.7"/>').join('');
    o += pad(19, 25, 7.6, 6.6, 0, '#5e8f4a') + pad(11, 15, 5.2, 6.4, -24, '#6a9a52') + pad(27, 13, 5.2, 6.4, 22, '#6a9a52') + pad(18, 7.5, 3.8, 4.4, 6, '#7aa862') + sp([[16, 24], [21, 22], [19, 28], [24, 27], [13, 26]]) + sp([[10, 14], [12, 18], [9, 11]]) + sp([[27, 12], [29, 16], [26, 9]]) + sp([[18, 6.4], [20, 9]]);
    [[11, 9], [27, 7], [18, 2.6], [20, 20]].slice(0, Math.max(f, 0)).forEach(([x, y]) => { o += E(x, y, 2.4, 2.9, '#c2386a') + CI(x - 0.7, y - 0.9, 0.6, '#f0a0c0') + '<path d="M' + (x - 1.4) + ' ' + (y - 3) + ' l1.4 -1 l1.4 1" stroke="#e8c840" stroke-width="1" fill="none"/>'; });
    if (!f) o += '<path d="M17 4 l1 -1.6 M19 4 l1 -1.6" stroke="#e8c840" stroke-width="1.1"/>';
  } else if (k === 'fruit') {
    o += L('M19 31 L19 20 M19 24 L12 18 M19 22 L27 16', '#6a4a2a', 2.6) + E(19, 11, 12, 8.4, '#3f8a3a') + E(10.5, 15, 6.2, 5, '#4a9a44') + E(27.5, 15, 6.2, 5, '#4a9a44') + E(19, 6.5, 6.6, 4, '#58a850') + '<path d="M9 12 q3 -2 6 0 M22 8 q3 -1 6 1 M18 15 q3 -1.4 6 0" stroke="#7fc86a" stroke-width="1.1" fill="none" stroke-linecap="round"/>';
    [[12, 15], [19, 9], [26, 15], [19, 15], [15, 11], [23, 11]].slice(0, 1 + f * 2).forEach(([x, y], i) => { const c = i % 2 ? '#e8a02c' : '#e06a2c'; o += E(x, y, 2.5, 2.7, c) + CI(x - 0.7, y - 0.8, 0.65, '#fbe090') + '<path d="M' + x + ' ' + (y - 2.6) + ' v-1" stroke="' + INK + '" stroke-width="0.9"/>'; });
    if (f >= 2) o += E(11, 31, 2.2, 1.8, '#e06a2c') + E(27, 31.4, 2, 1.7, '#e8a02c');
  } else if (k === 'cattail') {
    o += '<ellipse cx="19" cy="31" rx="15" ry="3" fill="#4a7a9a" opacity="0.55"/>';
    [[8, 14], [14, 7], [20, 11], [26, 5], [31, 13]].forEach(([x, y], i) => { o += L('M' + (x + (i % 2 ? 1 : -1)) + ' 31 Q' + x + ' 20 ' + x + ' ' + y, '#6a9a4a', 1.8); if (i < 2 + f) o += E(x, y + 2, 1.9, 5, '#6a4325') + '<path d="M' + x + ' ' + (y - 4) + ' v-3" stroke="#8a9a5a" stroke-width="1"/>'; });
  } else if (k === 'lichen') {
    const fr = (x, y, c, c2, n) => { let q = ''; for (let i = 0; i < n; i++) { const a = -Math.PI / 2 + (i - (n - 1) / 2) * 0.42, l = 5.4 + (i % 2) * 1.6, ex = x + Math.cos(a) * l, ey = y + Math.sin(a) * l, mx = x + Math.cos(a) * l * 0.55, my = y + Math.sin(a) * l * 0.55; q += '<path d="M' + x + ' ' + y + ' L' + mx.toFixed(1) + ' ' + my.toFixed(1) + ' L' + ex.toFixed(1) + ' ' + ey.toFixed(1) + ' M' + mx.toFixed(1) + ' ' + my.toFixed(1) + ' l' + (Math.cos(a + 0.7) * 2.6).toFixed(1) + ' ' + (Math.sin(a + 0.7) * 2.6).toFixed(1) + '" stroke="' + INK + '" stroke-width="3.1" fill="none" stroke-linecap="round" stroke-linejoin="round"/><path d="M' + x + ' ' + y + ' L' + mx.toFixed(1) + ' ' + my.toFixed(1) + ' L' + ex.toFixed(1) + ' ' + ey.toFixed(1) + ' M' + mx.toFixed(1) + ' ' + my.toFixed(1) + ' l' + (Math.cos(a + 0.7) * 2.6).toFixed(1) + ' ' + (Math.sin(a + 0.7) * 2.6).toFixed(1) + '" stroke="' + c + '" stroke-width="1.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>' + CI(ex, ey, 1.1, c2); } return q; };
    o += E(19, 29, 15, 4.2, '#6a6a66') + E(10, 27, 5.4, 3.2, '#82827c') + E(28, 28, 5, 3, '#7a7a74') + '<path d="M7 29 q3 -2 6 0 M24 29 q3 -1.6 6 0" stroke="#4a4a46" stroke-width="1" fill="none"/>';
    const cc = f ? '#a9bc8e' : '#8a9478', c2 = f ? '#c8d8a8' : '#9aa488';
    o += fr(11, 27, cc, c2, 3) + fr(24, 28, cc, c2, 3) + (f >= 1 ? fr(18, 28.6, cc, c2, 4) : '') + (f >= 2 ? fr(30, 27, cc, c2, 3) + fr(5, 28, cc, c2, 2) : '');
    if (f >= 3) o += CI(15, 20, 1, '#e08a3a') + CI(22, 19, 1, '#e08a3a') + CI(28, 21, 1, '#e08a3a');
  } else if (k === 'apple') {
    const g = f ? '#4a8a3a' : '#7a8a4a', g2 = f ? '#5fa04a' : '#8a9a58', D = (x, y, r, c) => CI(x, y, r, c, ' stroke="' + INK + '" stroke-width="0.9"');
    o += L('M19 31 L19 19 M19 23 L12 17 M19 22 L27 16', '#5a3f26', 2.8) + E(19, 11, 12.5, 8.4, g) + E(10.5, 15, 6.4, 5, g2) + E(27.5, 15, 6.4, 5, g2) + E(19, 6.5, 6.6, 4, g2);
    if (f === 0) o += '<path d="M12 8 q2 -3 4 -1 M21 7 q3 -2 5 1" stroke="#fff" stroke-width="1.6" fill="none" opacity="0.8"/>' + CI(14, 11, 0.9, '#fff') + CI(22, 10, 0.9, '#fff') + CI(26, 14, 0.9, '#fff');
    [[11, 15], [19, 10], [27, 15], [19, 16], [14, 10], [24, 10]].slice(0, [0, 2, 4, 6][f]).forEach(([x, y], i) => { o += D(x, y, 2.4, f === 1 ? '#9ac24a' : i % 3 ? '#d83a2e' : '#e8c03a') + CI(x - 0.7, y - 0.8, 0.6, '#ffe8d8') + '<path d="M' + x + ' ' + (y - 2.4) + ' v-1" stroke="' + INK + '" stroke-width="0.9"/>'; });
    if (f >= 2) o += D(11, 31, 2.2, '#d83a2e') + D(27, 31.4, 2, '#d83a2e');
  } else if (k === 'grape') {
    const lf = (x, y, r, c) => '<path d="M' + x + ' ' + (y - r) + ' q' + r + ' 0 ' + r + ' ' + r + ' q0 ' + r + ' -' + r + ' ' + r + ' q-' + r + ' 0 -' + r + ' -' + r + ' q0 -' + r + ' ' + r + ' -' + r + 'Z" fill="' + c + '" stroke="' + INK + '" stroke-width="1"/>';
    const lc = f ? '#5a9a3c' : '#8a9a50';
    o += '<path d="M8 31 L8 8 M30 31 L30 8 M8 10 H30" stroke="' + INK + '" stroke-width="3.4" fill="none"/><path d="M8 31 L8 8 M30 31 L30 8 M8 10 H30" stroke="#8a6a44" stroke-width="1.8" fill="none"/>';
    o += L('M12 31 Q10 20 16 14 Q22 8 28 10', '#6a4a2c', 1.8) + lf(14, 15, 3.6, lc) + lf(22, 11, 3.8, lc) + lf(27, 15, 3.4, lc) + lf(12, 23, 3.2, lc);
    const cl = (x, y, c) => { let q = '<path d="M' + x + ' ' + (y - 3) + ' v2" stroke="' + INK + '" stroke-width="1.2"/>'; [[0, 0], [-2, 1.6], [2, 1.6], [-1, 3.4], [1, 3.4], [0, 5]].forEach(([dx, dy]) => { q += CI(x + dx, y + dy, 1.6, c, ' stroke="' + INK + '" stroke-width="0.7"'); }); return q; };
    [[17, 18, '#5a3a8a'], [25, 19, '#6a4a9a'], [12, 27, '#5a3a8a']].slice(0, [0, 1, 2, 3][f]).forEach(([x, y, c]) => { o += cl(x, y, f === 1 ? '#9ac24a' : c); });
  } else if (k === 'banana') {
    const bl = (x, y, ex, ey, c) => '<path d="M' + x + ' ' + y + ' Q' + ((x + ex) / 2) + ' ' + (Math.min(y, ey) - 10) + ' ' + ex + ' ' + ey + ' Q' + ((x + ex) / 2 + (ex < x ? 2 : -2)) + ' ' + (Math.min(y, ey) + 3) + ' ' + x + ' ' + (y + 2.5) + 'Z" fill="' + c + '" stroke="' + INK + '" stroke-width="1.1"/>';
    const lc = f ? '#4a9a3a' : '#7aa04a', lc2 = f ? '#6ab84a' : '#8ab05a';
    o += L('M19 31 L19 13', '#7a9a4a', 4.6) + bl(19, 13, 2, 18, lc) + bl(19, 13, 36, 17, lc2) + bl(19, 12, 6, 7, lc2) + bl(19, 12, 32, 8, lc) + bl(19, 11, 19, 1, lc2);
    if (f >= 1) o += '<path d="M19 13 Q22 17 21 20" stroke="' + INK + '" stroke-width="1.4" fill="none"/>';
    if (f >= 2) { const bn = f >= 3 ? '#f2d03a' : '#9ac24a'; [[21, 19, 0], [23, 20, 10], [21, 22, 4], [23, 23, 12], [22, 25, 6]].slice(0, f >= 3 ? 5 : 3).forEach(([x, y, r]) => { o += '<path d="M' + x + ' ' + y + ' q3 1 2 5" stroke="' + INK + '" stroke-width="4.4" fill="none" stroke-linecap="round"/><path d="M' + x + ' ' + y + ' q3 1 2 5" stroke="' + bn + '" stroke-width="2.7" fill="none" stroke-linecap="round"/>'; }); if (f >= 3) o += E(22, 28.5, 1.8, 1.6, '#6a2a5a'); }
  } else if (k === 'date') {
    const fr = (ex, ey, c) => '<path d="M19 9 Q' + ((19 + ex) / 2) + ' ' + (Math.min(9, ey) - 5) + ' ' + ex + ' ' + ey + '" stroke="' + INK + '" stroke-width="3.4" fill="none" stroke-linecap="round"/><path d="M19 9 Q' + ((19 + ex) / 2) + ' ' + (Math.min(9, ey) - 5) + ' ' + ex + ' ' + ey + '" stroke="' + c + '" stroke-width="1.8" fill="none" stroke-linecap="round"/>';
    const lc = f ? '#4a8a3a' : '#8a9a50';
    o += '<path d="M17.5 31 Q18 20 18.2 10 L20.5 10 Q20.2 20 21 31Z" fill="#8a6a44" stroke="' + INK + '" stroke-width="1.3"/><path d="M18 14 h3 M18 18 h3 M18 22 h3.2 M18.4 26 h3" stroke="#5a4028" stroke-width="0.9"/>';
    o += fr(3, 13, lc) + fr(34, 13, lc) + fr(8, 5, lc) + fr(30, 5, lc) + fr(19, 1, lc) + fr(11, 16, lc) + fr(27, 16, lc);
    if (f >= 1) { const dc = f >= 3 ? '#c8742a' : f === 2 ? '#d8a43a' : '#a8b44a'; [[16.5, 13], [19, 14.4], [21.5, 13], [17.6, 16.2], [20.4, 16.2], [19, 18]].slice(0, 2 + f * 2).forEach(([x, y]) => { o += CI(x, y, 1.5, dc, ' stroke="' + INK + '" stroke-width="0.8"'); }); }
  } else if (k === 'olive') {
    const lc = f ? '#7a8e5a' : '#8a9a68', lc2 = f ? '#a4b27e' : '#a8b488';
    o += '<path d="M13 31 Q10 24 14 20 Q11 14 16 12 M25 31 Q28 24 23 20 Q27 14 22 12 M19 31 Q17 24 19 18" stroke="' + INK + '" stroke-width="5" fill="none" stroke-linecap="round"/><path d="M13 31 Q10 24 14 20 Q11 14 16 12 M25 31 Q28 24 23 20 Q27 14 22 12 M19 31 Q17 24 19 18" stroke="#8a7a62" stroke-width="3" fill="none" stroke-linecap="round"/>';
    o += E(19, 10, 12.5, 6.4, lc) + E(10, 14, 6.5, 4.4, lc2) + E(28, 14, 6.5, 4.4, lc2) + E(19, 6, 7.5, 3.6, lc2) + '<path d="M9 9 l3 -1 M16 5 l3 0 M24 8 l3 1 M12 14 l3 1 M24 14 l3 -1" stroke="#d8e0c0" stroke-width="1"/>';
    [[10, 16], [16, 12], [22, 12], [28, 16], [19, 8], [13, 9]].slice(0, [0, 2, 4, 6][f]).forEach(([x, y], i) => { o += E(x, y, 1.5, 1.9, f === 1 ? '#8aa44a' : i % 2 ? '#3a2a4a' : '#5a6a2a'); });
    if (f >= 2) o += E(10, 31, 1.6, 1.3, '#3a2a4a') + E(28, 31.4, 1.5, 1.2, '#5a6a2a');
  } else if (k === 'allium') {
    const bl = (x, ex, ey, c) => L('M' + x + ' 29 Q' + ((x + ex) / 2) + ' ' + (ey + 8) + ' ' + ex + ' ' + ey, c, 1.7);
    const gc = f ? '#5fa046' : '#8aa056';
    o += E(19, 30, 13, 3.4, '#6a4a30') + bl(14, 6, 8 - f, gc) + bl(16, 11, 4, gc) + bl(19, 19, 3 - f, gc) + bl(22, 27, 4, gc) + bl(24, 33, 9 - f, gc);
    if (f >= 2) o += L('M19 29 L19 12', '#7a9a4a', 1.6);
    if (f >= 2) { const fc = f >= 3 ? '#d8b4e6' : '#e8e0f0'; [[19, 8], [16.5, 10], [21.5, 10], [14.5, 8.6], [23.5, 8.6], [17, 6.4], [21, 6.4], [19, 11.4]].forEach(([x, y]) => { o += CI(x, y, 2.2, fc, ' stroke="' + INK + '" stroke-width="0.8"'); }); }
    o += E(15, 29, 3, 2.2, f >= 1 ? '#f2ecd8' : '#d8d2b8') + E(23.5, 29.6, 2.6, 2, f >= 1 ? '#f2ecd8' : '#d8d2b8') + '<path d="M13 31 l-2 2 M16 31 l1 2 M23 31 l-1 2 M25 31 l2 2" stroke="#8a7a5a" stroke-width="0.8"/>';
  } else if (k === 'carrot') {
    const fe = (x, h, c) => L('M' + x + ' 27 Q' + (x + (x < 19 ? -2 : 2)) + ' ' + (27 - h * 0.6) + ' ' + (x + (x < 19 ? -4 : 4)) + ' ' + (27 - h), c, 1.2);
    const gc = f ? '#4a9a3c' : '#8aa056';
    o += E(19, 30, 14, 3.8, '#6a4a30');
    [[13, 12], [16, 16], [19, 18], [22, 15], [25, 11], [10, 8], [28, 8]].slice(0, 3 + f * 2).forEach(([x, h]) => { o += fe(x, h, gc) + CI(x + (x < 19 ? -4 : 4), 27 - h, 1.2, gc); });
    if (f === 0) o += '<path d="M16 28 h6" stroke="#8a6a46" stroke-width="1.4"/>';
    if (f >= 1) o += E(19, 28.6, 2 + f * 0.6, 1.4 + f * 0.3, f >= 2 ? '#e6862a' : '#e8b86a') + '<path d="M17 28.2 q2 -1 4 0" stroke="#fbd49a" stroke-width="0.8" fill="none"/>';
    if (f >= 3) o += '<path d="M20 31 l-0.5 2.4 M18 31 l0.4 2" stroke="#e6862a" stroke-width="1.6"/>';
  } else if (k === 'gourd') {
    const lf = (x, y, r, c) => E(x, y, r, r * 0.75, c) + '<path d="M' + (x - r * 0.7) + ' ' + y + ' h' + (r * 1.4) + '" stroke="' + INK + '" stroke-width="0.7" opacity="0.5"/>';
    const lc = f ? '#4a9a3c' : '#8aa056';
    o += L('M4 28 Q12 22 19 27 Q26 30 34 24', '#5a8a3a', 1.6) + lf(9, 25, 4.6, lc) + lf(26, 28, 4.8, lc) + lf(32, 21, 3.6, lc) + lf(16, 21, 3.4, lc);
    if (f === 1) o += E(19, 28, 2.6, 2.2, '#9ac24a');
    if (f >= 2) { const gc = f >= 3 ? '#e08a2a' : '#c8b43a'; o += E(18, 27.4, 5.6, 4.6, gc) + '<path d="M14.4 24 q-1 3.4 0 7 M18 22.8 q-1 4.2 0 9 M21.6 24 q1 3.4 0 7" stroke="' + INK + '" stroke-width="0.8" fill="none" opacity="0.5"/>' + L('M18 22.8 q1 -2 3 -2.4', '#5a7a3a', 1.4); if (f >= 3) o += E(30, 29.6, 2.6, 2.2, '#c8b43a') + E(8, 30, 2, 1.7, '#e08a2a'); }
  }
  return o;
}

function rockSvg(v, m) {
  // faceted boulder lit from the upper left. m = material: 0 granite (speckled, blocky), 1 limestone (pale, rounded, bedded), 2 sandstone (warm, squat, banded strata), 3 basalt (dark, angular, columnar jointing), 4 schist (blue-grey, sheared slab, foliation + mica glints)
  m = m | 0;
  let P = [[[4, 24], [8, 9], [20, 3], [34, 6], [41, 19], [36, 35], [20, 38], [7, 33]], [[3, 20], [11, 5], [26, 2], [39, 11], [42, 27], [30, 37], [12, 37], [4, 31]], [[5, 26], [6, 12], [18, 4], [32, 4], [41, 15], [40, 30], [27, 38], [11, 36]]][v];
  if (m === 1) { for (let k = 0; k < 1; k++) { const Q = []; for (let i = 0; i < P.length; i++) { const a = P[i], b = P[(i + 1) % P.length]; Q.push([a[0] * 0.75 + b[0] * 0.25, a[1] * 0.75 + b[1] * 0.25], [a[0] * 0.25 + b[0] * 0.75, a[1] * 0.25 + b[1] * 0.75]); } P = Q; } }
  if (m === 2) P = P.map(([x, y]) => [x, 21 + (y - 21) * 0.84 + 3]);
  if (m === 3) P = P.map(([x, y]) => [22 + (x - 22) * 0.9, y]);
  if (m === 4) P = P.map(([x, y]) => [x + (21 - y) * 0.2 + (v - 1) * 1.5, 20 + (y - 20) * 0.9 + 2]);
  const base = [['#8d897e', '#948b7c', '#868a84'], ['#a79e8b', '#ada391', '#9e9684'], ['#b27a4c', '#a8744a', '#bd8556'], ['#45444c', '#3d3d45', '#4c4b52'], ['#6d7482', '#767c8a', '#656c7b']][m][v];
  const C = [22, 20], n = P.length, rr = mulberry(41 + v * 17 + m * 101), cid = 'q' + m + v;
  const pts = (a) => a.map((q) => q[0].toFixed(1) + ',' + q[1].toFixed(1)).join(' ');
  let o = '<ellipse cx="22" cy="37.5" rx="19" ry="3.2" fill="#000" opacity="0.28"/>';
  o += '<defs><clipPath id="' + cid + '"><polygon points="' + pts(P) + '"/></clipPath></defs>';
  o += poly(pts(P), INK, ' stroke="' + INK + '" stroke-width="2.4" stroke-linejoin="round"');
  const contrast = m === 3 ? 0.55 : 0.75;
  for (let i = 0; i < n; i++) {
    const a = P[i], b = P[(i + 1) % n], mx = (a[0] + b[0]) / 2 - C[0], my = (a[1] + b[1]) / 2 - C[1], mm = Math.hypot(mx, my) || 1, lit = (-mx * 0.62 - my * 0.78) / mm;
    const t = (lit + 1) / 2, col = t > 0.62 ? A.tint(base, (t - 0.62) * (m === 3 ? 0.5 : 0.7)) : darker(base, 0.52 + t * contrast);
    o += poly(pts([a, b, C]), col);
  }
  const top = P.map(([x, y]) => [C[0] + (x - C[0]) * 0.62 - 2.2, C[1] + (y - C[1]) * 0.55 - 4.2]);
  o += poly(pts(top), A.tint(base, m === 1 ? 0.12 : 0.2)) + poly(pts(top.map(([x, y]) => [C[0] + (x - C[0]) * 0.55 - 0.6, C[1] + (y - C[1]) * 0.5 - 0.8])), A.tint(base, m === 1 ? 0.2 : 0.34));
  if (m !== 1) for (let i = 0; i < n; i++) { const a = top[i], b = P[i]; o += '<path d="M' + a[0].toFixed(1) + ' ' + a[1].toFixed(1) + ' L' + b[0].toFixed(1) + ' ' + b[1].toFixed(1) + '" stroke="' + INK + '" stroke-width="1" opacity="0.32" fill="none"/>'; }
  o += '<path d="' + 'M' + pts(top).split(' ').join(' L') + ' Z' + '" fill="none" stroke="' + INK + '" stroke-width="1.1" opacity="0.5" stroke-linejoin="round"/>';
  // material texture, clipped to the outline
  let tx = '';
  const ln = (x1, y1, x2, y2, c, w, op) => '<path d="M' + x1.toFixed(1) + ' ' + y1.toFixed(1) + ' L' + x2.toFixed(1) + ' ' + y2.toFixed(1) + '" stroke="' + c + '" stroke-width="' + w + '" opacity="' + op + '" fill="none"/>';
  if (m === 0) {
    for (let i = 0; i < 34; i++) { const x = 5 + rr() * 34, y = 5 + rr() * 32, k = rr(); tx += '<circle cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="' + (0.5 + rr() * 0.9).toFixed(1) + '" fill="' + (k < 0.45 ? darker(base, 0.42) : k < 0.8 ? A.tint(base, 0.5) : '#b98f86') + '" opacity="0.75"/>'; }
    tx += '<path d="M' + (15 + v * 2) + ' 22 L' + (19 + v) + ' 27 L' + (17 + v) + ' 33 M28 14 L31 20 L29 23" stroke="' + INK + '" stroke-width="1.3" fill="none" stroke-linejoin="round" opacity="0.6"/>';
  } else if (m === 1) {
    for (let y = 9 + v; y < 38; y += 4.2 + rr() * 2) { const w = 0.6 + rr() * 0.8; tx += ln(2, y + rr() * 1.5, 42, y + rr() * 1.5 - 1, darker(base, 0.7), w, 0.5) + ln(2, y + 1.2, 42, y, A.tint(base, 0.5), 0.6, 0.5); }
    for (let i = 0; i < 4; i++) { const x = 8 + rr() * 26, y = 8 + rr() * 24; tx += '<ellipse cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" rx="' + (1 + rr() * 1.4).toFixed(1) + '" ry="0.8" fill="' + darker(base, 0.6) + '" opacity="0.6"/>'; }
  } else if (m === 2) {
    const cols = ['#d29a63', '#9a6038', '#c58450', '#7f4b2c', '#dfae78'];
    for (let y = 4, i = 0; y < 42; y += 3 + rr() * 3.5, i++) tx += '<path d="M0 ' + y.toFixed(1) + ' Q22 ' + (y - 1.6 + rr() * 3.2).toFixed(1) + ' 44 ' + (y + rr() * 2 - 1).toFixed(1) + ' L44 ' + (y + 2 + rr() * 2).toFixed(1) + ' Q22 ' + (y + 2 + rr() * 2).toFixed(1) + ' 0 ' + (y + 2).toFixed(1) + 'Z" fill="' + cols[i % 5] + '" opacity="0.5"/>';
    for (let i = 0; i < 12; i++) tx += '<circle cx="' + (6 + rr() * 32).toFixed(1) + '" cy="' + (8 + rr() * 28).toFixed(1) + '" r="0.6" fill="#f0d2a0" opacity="0.7"/>';
  } else if (m === 3) {
    for (let x = 6 + v; x < 42; x += 4.6 + rr() * 1.6) tx += ln(x, 2, x + (rr() - 0.5) * 1.6, 40, '#0e0e12', 1.1, 0.6) + ln(x + 1.1, 3, x + 1.1, 40, '#7a7a88', 0.5, 0.35);
    for (let i = 0; i < 8; i++) tx += '<circle cx="' + (6 + rr() * 32).toFixed(1) + '" cy="' + (8 + rr() * 28).toFixed(1) + '" r="' + (0.5 + rr() * 0.6).toFixed(1) + '" fill="#0a0a0d" opacity="0.8"/>';
    tx += '<path d="M12 18 L18 22 L16 28" stroke="#9aa0b8" stroke-width="0.7" fill="none" opacity="0.5"/>';
  } else {
    for (let k = -6; k < 16; k++) { const x0 = k * 4.4 + rr() * 1.2; tx += ln(x0, 42, x0 + 26, 0, rr() < 0.5 ? darker(base, 0.6) : A.tint(base, 0.35), 0.7 + rr() * 0.7, 0.55); }
    for (let i = 0; i < 10; i++) { const x = 6 + rr() * 32, y = 6 + rr() * 28; tx += '<path d="M' + (x - 1.2).toFixed(1) + ' ' + y.toFixed(1) + ' L' + (x + 1.2).toFixed(1) + ' ' + (y - 1.8).toFixed(1) + '" stroke="#e9eef6" stroke-width="0.9" opacity="0.8"/>'; }
  }
  o += '<g clip-path="url(#' + cid + ')">' + tx + '</g>';
  o += '<path d="M' + pts(P).split(' ').join(' L') + ' Z" fill="none" stroke="' + INK + '" stroke-width="1" opacity="0.5" stroke-linejoin="round"/>';
  for (let i = 0; i < 6; i++) { const x = 8 + rr() * 28, y = 8 + rr() * 26; o += '<circle cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="' + (0.4 + rr() * 0.5).toFixed(1) + '" fill="' + A.tint(base, 0.45) + '" opacity="0.5"/>'; }
  if (v !== 1 && m !== 3) o += '<path d="M10 30 q3 -3 6 -1 q2 -3 5 -1 q-1 4 -5 4 q-3 1 -6 -2Z" fill="#7f9a52" opacity="0.7"/><path d="M12 29.6 q2 -1.4 4 -0.4" stroke="#a8c070" stroke-width="1" fill="none" opacity="0.7"/>';
  return o;
}

const BIRD = '<ellipse cx="6" cy="8" rx="5.2" ry="3.6" fill="' + INK + '"/><circle cx="10.5" cy="5" r="3.6" fill="' + INK + '"/><path d="M14 4.4 L17 5.4 L14 6.4Z" fill="' + INK + '"/><path d="M2 8 L-1 11 L4 10Z" fill="' + INK + '"/><ellipse cx="6" cy="8" rx="4" ry="2.6" fill="#8a6a4a"/><circle cx="10.5" cy="5" r="2.6" fill="#a88866"/><path d="M13.5 4.7 L15.6 5.4 L13.5 6.1Z" fill="#d4a040"/><circle cx="11.2" cy="4.6" r="0.7" fill="' + INK + '"/><ellipse cx="5" cy="8.4" rx="2.6" ry="1.4" fill="#6a4e36"/>';

const bflySvg = (c1, c2) => '<ellipse cx="4.5" cy="4.5" rx="4" ry="3.3" fill="' + INK + '"/><ellipse cx="11.5" cy="4.5" rx="4" ry="3.3" fill="' + INK + '"/><ellipse cx="4.5" cy="4.5" rx="2.8" ry="2.2" fill="' + c1 + '"/><ellipse cx="11.5" cy="4.5" rx="2.8" ry="2.2" fill="' + c2 + '"/><rect x="7.3" y="1.5" width="1.4" height="7" rx="0.7" fill="' + INK + '"/>';

const FISH = '<ellipse cx="9" cy="5" rx="7" ry="3.2" fill="#14304a" opacity="0.7"/><path d="M15 5 L19 2 L19 8Z" fill="#14304a" opacity="0.7"/>';

const spr = (cls, w, h, inner) => '.' + cls + '{background-image:' + svgBg(w, h, inner) + ';background-size:100% 100%;background-repeat:no-repeat}';

const pg = (pts, f, ex) => '<polygon points="' + pts.map((q) => q[0].toFixed(1) + ',' + q[1].toFixed(1)).join(' ') + '" fill="' + f + '"' + (ex || '') + '/>';

function trunk3(cx, yb, yt, wb, wt, lit, shade, marks) {
  const L = [cx - wb / 2 - 2.5, yb + 1], R = [cx + wb / 2 + 2.5, yb + 1];
  const outline = [[cx - wt / 2, yt], [cx + wt / 2, yt], [cx + wb / 2, yb - 7], R, [cx + wb / 2 - 1.5, yb], [cx - wb / 2 + 1.5, yb], L, [cx - wb / 2, yb - 7]];
  let o = pg(outline, INK, ' stroke="' + INK + '" stroke-width="3.2" stroke-linejoin="round"');
  o += pg(outline, lit);
  o += pg([[cx + 0.5, yt], [cx + wt / 2, yt], [cx + wb / 2, yb - 7], R, [cx + wb / 2 - 1.5, yb], [cx + 0.5, yb]], shade);
  o += '<path d="M' + (cx - wb / 2 + 1) + ' ' + (yb - 7) + ' L' + (cx - wb / 2 - 1.5) + ' ' + (yb + 0.5) + ' M' + (cx + wb / 2 - 1) + ' ' + (yb - 7) + ' L' + (cx + wb / 2 + 1.5) + ' ' + (yb + 0.5) + '" stroke="' + INK + '" stroke-width="1" fill="none" opacity="0.5"/>';
  (marks || []).forEach(([dy, w]) => { o += '<path d="M' + (cx - w) + ' ' + (yb - dy) + ' l' + (w * 1.2) + ' 1.6" stroke="' + INK + '" stroke-width="1.3" stroke-linecap="round" fill="none" opacity="0.55"/>'; });
  return o;
}

function canopy3(lb, base, cx0, cy0) {
  const dk = darker(base, 0.7), hi = A.tint(base, 0.14), id = 'cp';
  let o = '<defs><clipPath id="' + id + '">' + lb.map(([cx, cy, r]) => '<circle cx="' + cx.toFixed(1) + '" cy="' + cy.toFixed(1) + '" r="' + r.toFixed(1) + '"/>').join('') + '</clipPath></defs>';
  lb.forEach(([cx, cy, r]) => { o += CI(cx, cy, r + 2, INK); });
  lb.forEach(([cx, cy, r]) => { o += CI(cx, cy, r, dk); });
  lb.forEach(([cx, cy, r]) => { o += CI(cx - 2, cy - 2, Math.max(2, r - 2.6), base); });
  // hard-edged facets across the whole crown: a lit upper-left plane and a shaded lower-right plane
  o += '<g clip-path="url(#' + id + ')">' + pg([[cx0 - 40, cy0 - 40], [cx0 + 6, cy0 - 40], [cx0 - 14, cy0 - 6], [cx0 - 40, cy0 + 6]], hi, ' opacity="0.55"') + pg([[cx0 + 40, cy0 - 14], [cx0 + 40, cy0 + 40], [cx0 - 22, cy0 + 40], [cx0 - 6, cy0 + 6], [cx0 + 18, cy0 - 2]], dk, ' opacity="0.5"') + pg([[cx0 - 30, cy0 + 40], [cx0 - 6, cy0 + 8], [cx0 + 8, cy0 + 40]], dk, ' opacity="0.35"') + '</g>';
  lb.forEach(([cx, cy, r], i) => { if (i % 3 === 1) o += CI(cx + r * 0.38, cy + r * 0.4, r * 0.16, dk); });
  return o;
}

function oak3(v, base, nl, R, sp) { const cx0 = 32, cy0 = 30; return trunk3(32, 78, 44, 10, 6.5, '#7a5538', '#553a26', [[12, 3], [22, 2.5]]) + canopy3(lobesOf(v, nl, R, sp, cx0, cy0), base, cx0, cy0) + '<ellipse cx="32" cy="47" rx="11" ry="3.2" fill="' + INK + '" opacity="0.38"/>'; }

function birch3(v, base) { const cx0 = 32, cy0 = 27; return trunk3(32, 78, 40, 7.5, 5, '#ece7d8', '#b9b3a2', [[10, 2.6], [17, 2.2], [26, 2.6], [33, 2]]) + canopy3(lobesOf(v + 1, 7, 9, 12.5, cx0, cy0), base, cx0, cy0) + '<ellipse cx="32" cy="43" rx="9" ry="2.6" fill="' + INK + '" opacity="0.38"/>'; }

function pine3(v) {
  const base = '#35563f', dk = darker(base, 0.7), tiers = [[63, 42, 21], [54, 31, 18], [45, 20, 14.5], [36, 9, 10.5]];
  let o = trunk3(32, 78, 56, 8, 6.5, '#7a5538', '#553a26', [[8, 2.4]]);
  tiers.forEach(([yb, yt, hw], i) => {
    const jag = [[32 - hw, yb], [32 - hw * 0.55, yb - 2.4], [32 - hw * 0.1, yb + 0.4], [32 + hw * 0.4, yb - 2.2], [32 + hw, yb]], apex = [32 + (v - 1) * 0.8, yt], c = A.tint(base, i * 0.09);
    const out = [apex, ...jag.slice().reverse()];
    o += pg(out, INK, ' stroke="' + INK + '" stroke-width="3.6" stroke-linejoin="round"');
    o += pg([apex, jag[0], jag[1], jag[2]], c) + pg([apex, jag[2], jag[3], jag[4]], dk);
    o += '<path d="M' + apex[0] + ' ' + (yt + 2) + ' L' + (32 - hw * 0.45) + ' ' + (yb - 3) + '" stroke="' + A.tint(base, 0.32) + '" stroke-width="1.4" opacity="0.8" fill="none"/>';
  });
  return o;
}

function palm3(v) {
  const lean = (v - 1) * 4, tx = 32 + lean, ty = 30;
  let o = '<path d="M32 78 Q' + (32 + lean * 0.2) + ' 54 ' + tx + ' ' + ty + '" stroke="' + INK + '" stroke-width="10.4" fill="none" stroke-linecap="round"/>';
  o += '<path d="M32 78 Q' + (32 + lean * 0.2) + ' 54 ' + tx + ' ' + ty + '" stroke="#8a6a48" stroke-width="6.6" fill="none" stroke-linecap="round"/>';
  o += '<path d="M33.4 78 Q' + (33.4 + lean * 0.2) + ' 54 ' + (tx + 1.4) + ' ' + ty + '" stroke="#a98a62" stroke-width="1.8" fill="none" stroke-linecap="round" opacity="0.8"/>';
  for (let i = 0; i < 6; i++) { const y = 72 - i * 7.5, x = 32 + lean * (1 - (y - ty) / 48) * 0.9; o += '<path d="M' + (x - 3.4).toFixed(1) + ' ' + y.toFixed(1) + ' l6.8 1.4" stroke="' + INK + '" stroke-width="1.3" opacity="0.6"/>'; }
  const fr = [[-168, 25], [-140, 28], [-108, 24], [-72, 24], [-40, 28], [-12, 25], [-90, 17]];
  const frond = (ang, len, c1, c2, w) => {
    const a = ang * Math.PI / 180, ex = tx + Math.cos(a) * len, ey = ty + Math.sin(a) * len * 0.8 + len * 0.45 * Math.abs(Math.cos(a)), mx = tx + Math.cos(a) * len * 0.55, my = ty + Math.sin(a) * len * 0.55 - 3, nx = -Math.sin(a), ny = Math.cos(a);
    const d = 'M' + tx + ' ' + ty + ' Q' + mx.toFixed(1) + ' ' + my.toFixed(1) + ' ' + ex.toFixed(1) + ' ' + ey.toFixed(1) + ' Q' + (mx + nx * w).toFixed(1) + ' ' + (my + ny * w + 2).toFixed(1) + ' ' + tx + ' ' + (ty + 1.5) + 'Z';
    return '<path d="' + d + '" fill="' + c1 + '" stroke="' + INK + '" stroke-width="2.4" stroke-linejoin="round"/><path d="M' + tx + ' ' + ty + ' Q' + mx.toFixed(1) + ' ' + my.toFixed(1) + ' ' + ex.toFixed(1) + ' ' + ey.toFixed(1) + '" stroke="' + c2 + '" stroke-width="1.2" fill="none"/>';
  };
  fr.forEach(([a, l], i) => { o += frond(a + (v - 1) * 4, l, i % 2 ? '#3f8a3e' : '#4c9a46', '#9cd06a', i % 2 ? 7 : -7); });
  o += CI(tx - 3, ty + 3, 2.6, '#6a4a2e', ' stroke="' + INK + '" stroke-width="1.2"') + CI(tx + 3, ty + 3.4, 2.6, '#6a4a2e', ' stroke="' + INK + '" stroke-width="1.2"');
  return o;
}

function acacia3(v) {
  const lean = (v - 1) * 3;
  let o = '<path d="M32 78 Q' + (31 + lean) + ' 58 ' + (30 + lean * 2) + ' 44 M' + (31 + lean) + ' 58 Q' + (40 + lean) + ' 50 ' + (44 + lean) + ' 40 M' + (30 + lean * 1.4) + ' 50 Q' + (22 + lean) + ' 44 ' + (18 + lean) + ' 38" stroke="' + INK + '" stroke-width="8.2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>';
  o += '<path d="M32 78 Q' + (31 + lean) + ' 58 ' + (30 + lean * 2) + ' 44 M' + (31 + lean) + ' 58 Q' + (40 + lean) + ' 50 ' + (44 + lean) + ' 40 M' + (30 + lean * 1.4) + ' 50 Q' + (22 + lean) + ' 44 ' + (18 + lean) + ' 38" stroke="#6e5238" stroke-width="4.4" fill="none" stroke-linecap="round" stroke-linejoin="round"/>';
  const lb = [[16 + lean, 34, 12, 6], [30 + lean, 27, 15, 7], [46 + lean, 32, 13, 6.4], [24 + lean, 38, 10, 4.4], [40 + lean, 37, 10, 4.4]];
  lb.forEach(([x, y, rx, ry]) => { o += '<ellipse cx="' + x + '" cy="' + y + '" rx="' + (rx + 1.6) + '" ry="' + (ry + 1.6) + '" fill="' + INK + '"/>'; });
  lb.forEach(([x, y, rx, ry], i) => { o += '<ellipse cx="' + x + '" cy="' + y + '" rx="' + rx + '" ry="' + ry + '" fill="' + (i % 2 ? '#7a9a3c' : '#6b8c36') + '"/><ellipse cx="' + (x - rx * 0.1) + '" cy="' + (y - ry * 0.35) + '" rx="' + rx * 0.7 + '" ry="' + ry * 0.45 + '" fill="#9ab850" opacity="0.8"/>'; });
  return o;
}

function bigleaf3(v) {
  let o = '';
  [[-34, 22], [18, 24], [-70, 19], [62, 19], [-4, 26]].forEach(([ang, len], i) => {
    const a = (ang - 90 + (v ? 8 : 0)) * Math.PI / 180, bx = 20, by = 29, ex = bx + Math.cos(a) * len, ey = by + Math.sin(a) * len * 0.8, nx = -Math.sin(a) * 5.5, ny = Math.cos(a) * 5.5, mx = (bx + ex) / 2, my = (by + ey) / 2;
    o += '<path d="M' + bx + ' ' + by + ' Q' + (mx + nx).toFixed(1) + ' ' + (my + ny).toFixed(1) + ' ' + ex.toFixed(1) + ' ' + ey.toFixed(1) + ' Q' + (mx - nx).toFixed(1) + ' ' + (my - ny).toFixed(1) + ' ' + bx + ' ' + by + 'Z" fill="' + (i % 2 ? '#3f9a3e' : '#4fae48') + '" stroke="' + INK + '" stroke-width="2.2" stroke-linejoin="round"/><path d="M' + bx + ' ' + by + ' L' + ex.toFixed(1) + ' ' + ey.toFixed(1) + '" stroke="#a6dc74" stroke-width="1.1"/>';
  });
  return o;
}

function liana3(v) {
  let o = '';
  [[10, 18], [20, 26], [30, 16], [15, 22]].forEach(([x, l], i) => { const sw = (i + v) % 2 ? 4 : -4; o += '<path d="M' + x + ' 0 Q' + (x + sw) + ' ' + l / 2 + ' ' + x + ' ' + l + '" stroke="' + INK + '" stroke-width="3.6" fill="none" stroke-linecap="round"/><path d="M' + x + ' 0 Q' + (x + sw) + ' ' + l / 2 + ' ' + x + ' ' + l + '" stroke="#4f7a38" stroke-width="1.8" fill="none" stroke-linecap="round"/>' + CI(x, l, 2, i % 2 ? '#e0709a' : '#e8c04c', ' stroke="' + INK + '" stroke-width="1"'); });
  return o;
}

function dead3(v) {
  const br = [[-13, 52, 2], [14, 47, 2], [-9, 38, 1], [10, 32, 1], [-5, 24 - v * 2, 0], [4, 17, 0]];
  let o = '', o2 = '';
  br.forEach(([dx, y, k]) => { const x = 32 + dx, tx = 32 + dx * 1.35, ty = y - 11 - k * 2, d = 'M32 ' + (y + 8) + ' L' + x + ' ' + y + ' L' + tx.toFixed(1) + ' ' + ty; o += '<path d="' + d + '" stroke="' + INK + '" stroke-width="5.2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>'; o2 += '<path d="' + d + '" stroke="#7a6048" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" fill="none"/>'; });
  return o + trunk3(32, 78, 20, 9, 4, '#7a6048', '#54402f', [[14, 3], [28, 2.6], [40, 2.4]]) + o2;
}

function willow3(v) {
  let o = trunk3(32, 78, 40, 9.5, 5.5, '#7a6a4e', '#54482f', [[12, 3], [22, 2.4]]);
  const lb = [[18, 30, 11, 8], [32, 24, 14, 9], [46, 30, 11, 8], [26, 31, 9, 6], [40, 31, 9, 6]];
  lb.forEach(([x, y, rx, ry]) => { o += '<ellipse cx="' + x + '" cy="' + y + '" rx="' + (rx + 1.8) + '" ry="' + (ry + 1.8) + '" fill="' + INK + '"/>'; });
  lb.forEach(([x, y, rx, ry], i) => { o += '<ellipse cx="' + x + '" cy="' + y + '" rx="' + rx + '" ry="' + ry + '" fill="' + (i % 2 ? '#8fb250' : '#7fa646') + '"/><ellipse cx="' + (x - rx * 0.12) + '" cy="' + (y - ry * 0.35) + '" rx="' + rx * 0.66 + '" ry="' + ry * 0.4 + '" fill="#b4d070" opacity="0.8"/>'; });
  for (let i = 0; i < 9; i++) { const x = 11 + i * 5.2 + (v - 1) * 1.2, len = 20 + ((i * 7 + v * 3) % 9), sw = (i % 2 ? 3 : -3); const d = 'M' + x + ' 32 Q' + (x + sw) + ' ' + (32 + len * 0.55) + ' ' + (x + sw * 0.4) + ' ' + (32 + len); o += '<path d="' + d + '" stroke="' + INK + '" stroke-width="4" fill="none" stroke-linecap="round"/>'; }
  for (let i = 0; i < 9; i++) { const x = 11 + i * 5.2 + (v - 1) * 1.2, len = 20 + ((i * 7 + v * 3) % 9), sw = (i % 2 ? 3 : -3); const d = 'M' + x + ' 32 Q' + (x + sw) + ' ' + (32 + len * 0.55) + ' ' + (x + sw * 0.4) + ' ' + (32 + len); o += '<path d="' + d + '" stroke="' + (i % 2 ? '#8cb04e' : '#a3c25c') + '" stroke-width="2" fill="none" stroke-linecap="round"/>'; }
  return o + '<ellipse cx="32" cy="79" rx="13" ry="3" fill="' + INK + '" opacity="0.32"/>';
}

function krum3(v) { return '<g transform="translate(32 78) skewX(' + (-10 - v * 3) + ') scale(0.72 0.58) translate(-32 -78)">' + pine3(v) + '</g>'; }

function joshua3(v) {
  const lean = (v - 1) * 2.4, arms = [[-12 + lean, 50, -14 + lean, 30], [12 + lean, 44, 16 + lean, 26], [1 + lean, 34, 0 + lean, 15 - v * 2]];
  let o = '', o2 = '';
  const tuft = (x, y, r) => { let t = ''; for (let a = 0; a < 11; a++) { const an = (a / 11) * Math.PI * 2, ex = x + Math.cos(an) * r, ey = y + Math.sin(an) * r * 0.85; t += '<path d="M' + x + ' ' + y + ' L' + ex.toFixed(1) + ' ' + ey.toFixed(1) + '" stroke="' + INK + '" stroke-width="3.4" stroke-linecap="round"/>'; } for (let a = 0; a < 11; a++) { const an = (a / 11) * Math.PI * 2, ex = x + Math.cos(an) * r, ey = y + Math.sin(an) * r * 0.85; t += '<path d="M' + x + ' ' + y + ' L' + ex.toFixed(1) + ' ' + ey.toFixed(1) + '" stroke="' + (a % 2 ? '#7a8a3c' : '#92a24a') + '" stroke-width="1.7" stroke-linecap="round"/>'; } return t + CI(x, y, 2.4, '#6a7a34'); };
  arms.forEach(([bx, by, tx, ty]) => { const d = 'M' + (32 + (bx > 0 ? 1 : bx < -2 ? -1 : 0)) + ' ' + (by + 8) + ' Q' + (32 + bx * 0.9) + ' ' + (by - 2) + ' ' + (32 + tx) + ' ' + (ty + 6); o += '<path d="' + d + '" stroke="' + INK + '" stroke-width="7.4" fill="none" stroke-linecap="round"/>'; o2 += '<path d="' + d + '" stroke="#8a7458" stroke-width="3.8" fill="none" stroke-linecap="round"/>'; });
  let t = ''; arms.forEach(([bx, by, tx, ty]) => { t += tuft(32 + tx, ty + 4, 9.4); });
  return trunk3(32, 78, 46, 9.4, 6, '#8a7458', '#5e4c38', [[14, 2.8], [28, 2.4], [40, 2.4]]) + o + o2 + t + '<ellipse cx="32" cy="79" rx="11" ry="2.6" fill="' + INK + '" opacity="0.32"/>';
}

function baobab3(v) {
  const tr = 'M17 78 Q9 62 16 46 Q18 38 24 34 L40 34 Q46 38 48 46 Q55 62 47 78Z';
  let o = '<path d="' + tr + '" fill="' + INK + '" stroke="' + INK + '" stroke-width="4" stroke-linejoin="round"/><path d="' + tr + '" fill="#97805e"/>';
  o += '<path d="M33 34 L40 34 Q46 38 48 46 Q55 62 47 78 L35 78 Q40 56 33 34Z" fill="#7a664a"/><path d="M22 40 Q14 58 20 74" stroke="#b49c78" stroke-width="2.2" fill="none" stroke-linecap="round" opacity="0.8"/><path d="M26 48 q3 2 6 0 M24 60 q4 2 8 0 M36 52 q3 2 6 0" stroke="#5e4c38" stroke-width="1.3" fill="none" opacity="0.7"/>';
  const br = [[-10, 26 - v, -16, 20], [3, 25, 2, 15 + v], [12, 28, 18, 21]]; let o2 = '', o3 = '';
  br.forEach(([x0, y0, x1, y1]) => { const d = 'M' + (32 + x0 * 0.5) + ' 36 Q' + (32 + x0) + ' ' + y0 + ' ' + (32 + x1) + ' ' + y1; o2 += '<path d="' + d + '" stroke="' + INK + '" stroke-width="6.4" fill="none" stroke-linecap="round"/>'; o3 += '<path d="' + d + '" stroke="#97805e" stroke-width="3" fill="none" stroke-linecap="round"/>'; });
  const lb = [[16, 19, 8], [32, 13 + v, 9], [49, 20, 8], [24, 16, 6], [41, 15, 6]]; let c = '';
  lb.forEach(([x, y, r]) => { c += CI(x, y, r + 1.8, INK); }); lb.forEach(([x, y, r], i) => { c += CI(x, y, r, i % 2 ? '#6e8f3a' : '#7c9c42') + CI(x - r * 0.2, y - r * 0.3, r * 0.5, '#9cba5a'); });
  return o + o2 + o3 + c + '<ellipse cx="32" cy="79" rx="17" ry="3.2" fill="' + INK + '" opacity="0.34"/>';
}

const blossom3 = (lb, c1, c2) => lb.map(([cx, cy, r], i) => CI(cx + r * 0.2, cy - r * 0.25, 1.9, i % 2 ? c1 : c2) + CI(cx - r * 0.45, cy + r * 0.2, 1.5, i % 2 ? c2 : c1)).join('');

function bare3(v, birch) {
  const cy = birch ? 40 : 44, L = birch ? 26 : 30, br = [[-0.95, 1], [-0.55, 1.15], [-0.2, 1.25], [0.15, 1.2], [0.5, 1.15], [0.95, 1], [-1.25, 0.7], [1.25, 0.7], [0.0, 0.8]];
  let o = '', o2 = '', sn = '';
  br.forEach(([a, k], i) => {
    const l = L * k * (0.9 + 0.2 * ((i + v) % 3) / 2), x = 32 + Math.sin(a) * l, y = cy - Math.cos(a) * l * 0.95, mx = 32 + Math.sin(a) * l * 0.5, my = cy - Math.cos(a) * l * 0.5;
    const fx = mx + Math.sin(a + 0.8) * 9, fy = my - Math.cos(a + 0.8) * 9, fx2 = mx + Math.sin(a - 0.8) * 8, fy2 = my - Math.cos(a - 0.8) * 8;
    const d = 'M32 ' + (cy + 3) + ' L' + mx.toFixed(1) + ' ' + my.toFixed(1) + ' L' + x.toFixed(1) + ' ' + y.toFixed(1) + ' M' + mx.toFixed(1) + ' ' + my.toFixed(1) + ' L' + fx.toFixed(1) + ' ' + fy.toFixed(1) + ' M' + mx.toFixed(1) + ' ' + my.toFixed(1) + ' L' + fx2.toFixed(1) + ' ' + fy2.toFixed(1);
    o += '<path d="' + d + '" stroke="' + INK + '" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>';
    o2 += '<path d="' + d + '" stroke="' + (birch ? '#cfc9b8' : '#7a5538') + '" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" fill="none"/>';
    sn += '<path d="M' + mx.toFixed(1) + ' ' + (my - 1.4).toFixed(1) + ' L' + x.toFixed(1) + ' ' + (y - 1.4).toFixed(1) + '" stroke="#f4f8fc" stroke-width="1.6" stroke-linecap="round" fill="none"/>';
  });
  return (birch ? trunk3(32, 78, 40, 7.5, 5, '#ece7d8', '#b9b3a2', [[10, 2.6], [17, 2.2], [26, 2.6], [33, 2]]) : trunk3(32, 78, 44, 10, 6.5, '#7a5538', '#553a26', [[12, 3], [22, 2.5]])) + o + o2 + sn + '<ellipse cx="32" cy="78" rx="12" ry="2.4" fill="#f4f8fc" opacity="0.9"/>';
}

const snowPine3 = (v) => { let o = pine3(v); [[63, 42, 21], [54, 31, 18], [45, 20, 14.5], [36, 9, 10.5]].forEach(([yb, yt, hw]) => { o += pg([[32, yt - 0.5], [32 - hw * 0.62, yt + (yb - yt) * 0.55], [32 - hw * 0.2, yt + (yb - yt) * 0.5], [32 + hw * 0.1, yt + (yb - yt) * 0.62], [32 + hw * 0.62, yt + (yb - yt) * 0.55]], '#f4f8fc', ' stroke="' + INK + '" stroke-width="1" stroke-linejoin="round"'); }); return o; };

function cactus3(v) {
  const g = '#5f8a4a', gd = '#3f6a38', gl = '#86b266';
  const col = (x, w, yt, yb) => { const out = [[x - w / 2, yb], [x - w / 2, yt + 4], [x - w / 4, yt], [x + w / 4, yt], [x + w / 2, yt + 4], [x + w / 2, yb]]; return pg(out, INK, ' stroke="' + INK + '" stroke-width="3.4" stroke-linejoin="round"') + pg(out, g) + pg([[x + 0.5, yt], [x + w / 4, yt], [x + w / 2, yt + 4], [x + w / 2, yb], [x + 0.5, yb]], gd) + '<path d="M' + (x - w / 4) + ' ' + (yt + 6) + ' V' + (yb - 3) + '" stroke="' + gl + '" stroke-width="1.4" stroke-linecap="round"/>'; };
  let o = '<ellipse cx="20" cy="60" rx="13" ry="3.4" fill="' + INK + '" opacity="0.35"/>';
  const arm = (x0, y0, dir, up) => '<path d="M' + x0 + ' ' + y0 + ' H' + (x0 + dir * 11) + ' V' + (y0 - up) + '" stroke="' + INK + '" stroke-width="9.4" stroke-linecap="round" stroke-linejoin="round" fill="none"/><path d="M' + x0 + ' ' + y0 + ' H' + (x0 + dir * 11) + ' V' + (y0 - up) + '" stroke="' + g + '" stroke-width="6" stroke-linecap="round" stroke-linejoin="round" fill="none"/>';
  o += arm(20, 42, -1, 16 + v * 3) + arm(20, 32, 1, 14 + (v ^ 1) * 4) + col(20, 13, 6, 60);
  o += '<circle cx="20" cy="6" r="2.4" fill="#e0709a" stroke="' + INK + '" stroke-width="1"/>';
  return o + '<path d="M17 12 l1 -0 M22 20 l1 0 M17 30 l1 0 M22 42 l1 0 M17 50 l1 0" stroke="' + INK + '" stroke-width="1.2" opacity="0.5"/>';
}

function reeds3(v) {
  let o = '<ellipse cx="20" cy="41" rx="14" ry="3" fill="' + INK + '" opacity="0.3"/>';
  [[-9, 22, -4], [-5, 30, -2], [-1, 36, 1], [3, 32, 3], [7, 26, 5], [11, 20, 6]].forEach(([dx, h, lean], i) => {
    const x0 = 20 + dx, d = 'M' + x0 + ' 41 Q' + (x0 + lean * 0.4) + ' ' + (41 - h * 0.55) + ' ' + (x0 + lean) + ' ' + (41 - h);
    o += '<path d="' + d + '" stroke="' + INK + '" stroke-width="4.6" stroke-linecap="round" fill="none"/><path d="' + d + '" stroke="' + (i % 2 ? '#7a9a48' : '#8fae58') + '" stroke-width="2.2" stroke-linecap="round" fill="none"/>';
    if ((i + v) % 2 === 0) o += '<rect x="' + (x0 + lean - 1.8) + '" y="' + (41 - h - 5) + '" width="3.6" height="9" rx="1.8" fill="#6a4a30" stroke="' + INK + '" stroke-width="1.2"/>';
  });
  return o;
}

function stump3() {
  const out = [[6, 12], [8, 24], [14, 27], [24, 28], [30, 25], [31, 13]];
  return '<ellipse cx="19" cy="27" rx="15" ry="3.4" fill="' + INK + '" opacity="0.35"/>' + pg(out, INK, ' stroke="' + INK + '" stroke-width="3.4" stroke-linejoin="round"') + pg(out, '#7a5538') + pg([[19, 14], [31, 13], [30, 25], [24, 28], [19, 28]], '#553a26') + '<path d="M10 24 l1 3 M26 22 l0 5" stroke="' + INK + '" stroke-width="1.1" opacity="0.5"/>'
    + '<ellipse cx="18.5" cy="12" rx="12.5" ry="5.4" fill="' + INK + '"/><ellipse cx="18.5" cy="11.6" rx="11" ry="4.4" fill="#c9a06c"/><ellipse cx="18.5" cy="11.6" rx="7.2" ry="2.8" fill="none" stroke="#9a7448" stroke-width="1"/><ellipse cx="18.5" cy="11.6" rx="3.4" ry="1.3" fill="none" stroke="#9a7448" stroke-width="1"/>';
}

function log3() {
  return '<ellipse cx="30" cy="25" rx="26" ry="3" fill="' + INK + '" opacity="0.3"/><rect x="6" y="6" width="46" height="18" rx="5" fill="' + INK + '"/><rect x="8" y="8" width="42" height="14" rx="3.5" fill="#7a5538"/><rect x="8" y="16" width="42" height="6" rx="3" fill="#553a26"/><path d="M16 9 l3 4 M26 8 l2 3 M36 10 l3 3 M22 17 l2 3 M42 17 l2 3" stroke="' + INK + '" stroke-width="1.2" stroke-linecap="round" opacity="0.5"/>'
    + '<ellipse cx="8" cy="15" rx="5.4" ry="9" fill="' + INK + '"/><ellipse cx="8.4" cy="15" rx="4" ry="7.4" fill="#c9a06c"/><ellipse cx="8.6" cy="15" rx="2.2" ry="4.2" fill="none" stroke="#9a7448" stroke-width="1"/><path d="M40 8 l4 -4 l3 1 l-2 5 z" fill="#553a26" stroke="' + INK + '" stroke-width="1.4" stroke-linejoin="round"/>';
}

function fern3(v) {
  let o = '<ellipse cx="20" cy="27" rx="13" ry="2.8" fill="' + INK + '" opacity="0.3"/>';
  const fr = [[-1.3, 15], [-0.8, 19], [-0.3, 21], [0.25, 21], [0.8, 19], [1.3, 15]];
  fr.forEach(([a, l], i) => { const x = 20 + Math.sin(a) * l, y = 27 - Math.cos(a) * l * 0.85, d = 'M20 27 Q' + (20 + Math.sin(a) * l * 0.5) + ' ' + (27 - Math.cos(a) * l * 0.9) + ' ' + x.toFixed(1) + ' ' + y.toFixed(1); o += '<path d="' + d + '" stroke="' + INK + '" stroke-width="5" stroke-linecap="round" fill="none"/><path d="' + d + '" stroke="' + (i % 2 ? '#4f8a4a' : '#62a05a') + '" stroke-width="2.6" stroke-linecap="round" fill="none"/>'; });
  return o;
}

function shroom3(v) {
  const cap = v ? '#c2503c' : '#a8825a', dot = v ? '#f4ece0' : '#d8c4a0';
  const one = (x, y, k) => '<rect x="' + (x - 1.6 * k) + '" y="' + (y - 1) + '" width="' + (3.2 * k) + '" height="' + (6 * k) + '" rx="1.4" fill="#e8dcc4" stroke="' + INK + '" stroke-width="1.3"/><path d="M' + (x - 6 * k) + ' ' + (y + 1) + ' Q' + x + ' ' + (y - 10 * k) + ' ' + (x + 6 * k) + ' ' + (y + 1) + ' Z" fill="' + cap + '" stroke="' + INK + '" stroke-width="1.6" stroke-linejoin="round"/><circle cx="' + (x - 1.8 * k) + '" cy="' + (y - 3 * k) + '" r="' + (1.1 * k) + '" fill="' + dot + '"/><circle cx="' + (x + 2.2 * k) + '" cy="' + (y - 2 * k) + '" r="' + (0.9 * k) + '" fill="' + dot + '"/>';
  return '<ellipse cx="15" cy="21" rx="12" ry="2.4" fill="' + INK + '" opacity="0.3"/>' + one(9, 17, 0.9) + one(20, 18, 1.15) + one(15, 20, 0.7);
}

const SPRITE_CSS = (() => {
  let c = '';
  for (let v = 0; v < 3; v++) {
    c += spr('t-oak' + v, 64, 84, oak3(v, ['#33603a', '#2d5534', '#3b6a3a'][v], 7 + v, 11.5, 16.5));
    c += spr('t-birch' + v, 64, 84, birch3(v, ['#6f9a4c', '#648e46', '#7aa456'][v]));
    c += spr('t-pine' + v, 64, 84, pine3(v));
    c += spr('t-palm' + v, 64, 84, palm3(v));
    c += spr('t-acacia' + v, 64, 84, acacia3(v));
    c += spr('t-willow' + v, 64, 84, willow3(v)) + spr('t-krum' + v, 64, 84, krum3(v)) + spr('t-joshua' + v, 64, 84, joshua3(v)) + spr('t-baobab' + v, 64, 84, baobab3(v));
    for (let m = 0; m < 5; m++) c += spr('rk' + m + v, 44, 40, rockSvg(v, m));
  }
  for (let v = 0; v < 2; v++) c += spr('t-dead' + v, 64, 84, dead3(v));
  // seasons: 0 spring (blossom), 2 autumn (turned leaves), 3 winter (bare snowy crowns, snow-laden pines); summer is the base look
  for (let v = 0; v < 3; v++) {
    const ob = ['#4f8a42', '#46803c', '#58944a'][v], oa = ['#c0662a', '#b9832a', '#a8462c'][v], ba = ['#e0b030', '#d89a2c', '#e6c24a'][v], bb = ['#86b258', '#7cac52', '#92ba62'][v];
    c += spr('q0.t-oak' + v, 64, 84, oak3(v, ob, 7 + v, 11.5, 16.5).replace('</svg>', '') + blossom3(lobesOf(v, 7 + v, 11.5, 16.5, 32, 30), '#f4c8d8', '#fbeef2'));
    c += spr('q0.t-birch' + v, 64, 84, birch3(v, bb) + blossom3(lobesOf(v + 1, 7, 9, 12.5, 32, 27), '#f6f0c8', '#ffffff'));
    c += spr('q2.t-oak' + v, 64, 84, oak3(v, oa, 7 + v, 11.5, 16.5));
    c += spr('q2.t-birch' + v, 64, 84, birch3(v, ba));
    c += spr('q3.t-oak' + v, 64, 84, bare3(v, false));
    c += spr('q3.t-birch' + v, 64, 84, bare3(v, true));
    c += spr('q3.t-pine' + v, 64, 84, snowPine3(v));
  }
  c += spr('p-bigleaf0', 40, 30, bigleaf3(0)) + spr('p-bigleaf1', 40, 30, bigleaf3(1)) + spr('p-liana0', 40, 30, liana3(0)) + spr('p-liana1', 40, 30, liana3(1));
  c += spr('p-cactus0', 40, 64, cactus3(0)) + spr('p-cactus1', 40, 64, cactus3(1)) + spr('p-reeds0', 40, 46, reeds3(0)) + spr('p-reeds1', 40, 46, reeds3(1)) + spr('p-stump', 38, 30, stump3()) + spr('p-log', 60, 28, log3()) + spr('p-fern0', 40, 30, fern3(0)) + spr('p-fern1', 40, 30, fern3(1)) + spr('p-shroom0', 30, 24, shroom3(0)) + spr('p-shroom1', 30, 24, shroom3(1));
  for (let f = 0; f < 4; f++) c += spr('bush' + f, 38, 34, bushSvg(f));
  for (const fk of Object.keys(KIND)) if (fk !== 'berry') for (let f = 0; f < 4; f++) c += spr('fg-' + fk + f, 38, 34, forageSvg(fk, f));
  for (const sk of Object.keys(SEA)) c += spr('fs-' + sk, 42, 20, fishSvg(sk));
  for (const mk of Object.keys(MAR)) c += spr('ms-' + mk, MAR[mk].vw, MAR[mk].vh, marSvg(mk));
  for (const mk of Object.keys(MARP)) c += spr('mp-' + mk, 40, 40, marpSvg(mk));
  c += spr('bird', 18, 12, BIRD) + spr('bfly0', 16, 9, bflySvg('#e0956a', '#e8c04c')) + spr('bfly1', 16, 9, bflySvg('#9cc4ee', '#ede9e1')) + spr('fish', 20, 10, FISH);
  // fire: three separately animated layers (ink-outlined multi-tongue outer, orange body in ::before, yellow-white core in ::after)
  const FP = ['M4 34 C4 26 8 22 11 20 L29 20 C32 22 36 26 36 34 C36 42 29 48 20 48 C11 48 4 42 4 34Z', 'M8 27 C3 21 4 12 8 5 C9 14 15 18 15 27Z', 'M32 27 C37 22 36 14 31 8 C30 16 25 19 25 27Z', 'M20 1 C23 11 31 17 30 31 L10 31 C9 17 17 11 20 1Z'];
  const FIRE = (v) => {
    const m = v ? 'translate(40 0) scale(-1 1)' : '';
    return '<g transform="' + m + '">' + FP.map((d) => '<path d="' + d + '" fill="' + INK + '" stroke="' + INK + '" stroke-width="5" stroke-linejoin="round"/>').join('') + FP.map((d, i) => '<path d="' + d + '" fill="' + (i === 3 ? '#e5532f' : '#d6452a') + '"/>').join('') + '<path d="M6 36 C6 41 12 46 20 46" fill="none" stroke="#a82f1f" stroke-width="2" stroke-linecap="round" opacity="0.7"/></g>';
  };
  const FIRE_M = (v) => '<g transform="' + (v ? 'translate(40 0) scale(-1 1)' : '') + '"><path d="M20 12 C23 21 28 26 28 35 C28 42 25 46 20 46 C15 46 12 42 12 35 C12 28 17 21 20 12Z" fill="#f59a3a"/><path d="M11 33 C8 29 9 24 12 20 C13 26 15 28 14 33Z" fill="#f59a3a"/><path d="M29 33 C32 29 31 24 28 20 C27 26 25 28 26 33Z" fill="#f59a3a"/></g>';
  const FIRE_C = '<path d="M20 25 C22 31 25 34 25 39 C25 43 23 45 20 45 C17 45 15 43 15 39 C15 34 18 31 20 25Z" fill="#ffd95a"/><path d="M20 34 C21.5 37 23 38.5 23 41 C23 43 21.8 44 20 44 C18.2 44 17 43 17 41 C17 38.5 18.5 37 20 34Z" fill="#fff7d0"/>';
  c += spr('fire0', 40, 50, FIRE(0)) + spr('fire1', 40, 50, FIRE(1));
  // walls seen from above: a square block that grows arms towards neighbouring wall tiles, so runs, corners, junctions and rings join into one shape. The top is the material's colour with its grain; a lighter strip along any south-facing edge is the visible face of the wall; dark outline; a separate soft shadow sprite goes under it, cast to the south-east
  { const WCOL = { wattle: ['#b8a27c', '#d9c6a0', '#8f7b57'], timber: ['#6f5238', '#9a7a56', '#4a3524'], stone: ['#5f5c58', '#8f8c86', '#3f3d3a'] };
    const union = (m, g) => { const r = 3, o = []; o.push('<rect x="' + (10 - g) + '" y="' + (10 - g) + '" width="' + (28 + 2 * g) + '" height="' + (28 + 2 * g) + '" rx="' + (r + g) + '"/>'); if (m & 1) o.push('<rect x="' + (10 - g) + '" y="-6" width="' + (28 + 2 * g) + '" height="' + (30 + g) + '"/>'); if (m & 2) o.push('<rect x="' + (24 - g) + '" y="' + (10 - g) + '" width="' + (30 + g) + '" height="' + (28 + 2 * g) + '"/>'); if (m & 4) o.push('<rect x="' + (10 - g) + '" y="' + (24 - g) + '" width="' + (28 + 2 * g) + '" height="' + (30 + g) + '"/>'); if (m & 8) o.push('<rect x="-6" y="' + (10 - g) + '" width="' + (30 + g) + '" height="' + (28 + 2 * g) + '"/>'); return o.join(''); };
    for (let m = 0; m < 16; m++) c += spr('wsh' + m, 48, 48, '<g fill="#000" opacity="0.4" transform="translate(5 6)">' + union(m, 1) + '</g>');
    const grain = (mat, m, rr) => { let t = ''; if (mat === 'stone') { for (let i = 0; i < 12; i++) t += '<circle cx="' + (4 + rr() * 40).toFixed(1) + '" cy="' + (4 + rr() * 40).toFixed(1) + '" r="' + (0.6 + rr()).toFixed(1) + '" fill="' + (rr() < 0.5 ? '#4b4945' : '#787570') + '" opacity="0.7"/>'; t += '<path d="M14 22 l6 5 l-3 6 M30 12 l4 6 l-2 5" stroke="#3a3835" stroke-width="1.1" fill="none" opacity="0.6"/>'; } else if (mat === 'timber') { for (let i = 0; i < 6; i++) t += '<path d="M' + (12 + i * 5) + ' 4 v40" stroke="#4a3524" stroke-width="0.9" opacity="' + (0.35 + 0.3 * (i % 2)) + '"/>'; if ((m & 10) && !(m & 5)) for (let i = 0; i < 6; i++) t += '<path d="M4 ' + (12 + i * 5) + ' h40" stroke="#4a3524" stroke-width="0.9" opacity="' + (0.35 + 0.3 * (i % 2)) + '"/>'; t += '<circle cx="19" cy="20" r="1.3" fill="#3b2a1b" opacity="0.7"/><circle cx="31" cy="29" r="1.3" fill="#3b2a1b" opacity="0.7"/>'; } else { for (let i = 0; i < 9; i++) t += '<circle cx="' + (6 + rr() * 36).toFixed(1) + '" cy="' + (6 + rr() * 36).toFixed(1) + '" r="' + (0.8 + rr() * 1.4).toFixed(1) + '" fill="#8f7b57" opacity="0.35"/>'; t += '<path d="M8 16 q8 3 16 0 t16 0 M8 32 q8 -3 16 0 t16 0" stroke="#7e6b4b" stroke-width="1" fill="none" opacity="0.5"/>'; } return t; };
    for (const mat of ['wattle', 'timber', 'stone']) for (let m = 0; m < 16; m++) for (const w of [0, 1]) { const [base, lite, dark] = WCOL[mat], rr = mulberry(7 + m * 13 + w * 101 + mat.length), id = 'wc' + mat[0] + m + w;
      let o = '<defs><clipPath id="' + id + '">' + union(m, 0) + '</clipPath></defs>';
      o += '<g fill="' + INK + '">' + union(m, 3) + '</g><g fill="' + base + '">' + union(m, 0) + '</g><g clip-path="url(#' + id + ')">' + grain(mat, m, rr);
      // the face: along every south-facing edge of the block a lighter strip, the wall's side catching the light; the arms to east and west carry it too, the arm to the south does not
      if (!(m & 4)) o += '<rect x="10" y="31" width="28" height="7" fill="' + lite + '"/><rect x="10" y="31" width="28" height="1.5" fill="' + dark + '" opacity="0.7"/>'; if (m & 2) o += '<rect x="24" y="31" width="30" height="7" fill="' + lite + '"/><rect x="24" y="31" width="30" height="1.5" fill="' + dark + '" opacity="0.7"/>'; if (m & 8) o += '<rect x="-6" y="31" width="30" height="7" fill="' + lite + '"/><rect x="-6" y="31" width="30" height="1.5" fill="' + dark + '" opacity="0.7"/>';
      o += '<g fill="' + lite + '" opacity="0.35" transform="translate(0 -1.5)">' + union(m, 0) + '</g><g fill="' + base + '">' + union(m, 0).replace(/<rect /g, '<rect transform="translate(0 1.5)" ') + '</g>' + (!(m & 4) ? '<rect x="10" y="31" width="28" height="7" fill="' + lite + '"/><rect x="10" y="31" width="28" height="1.5" fill="' + dark + '" opacity="0.7"/>' : '') + ((m & 2) ? '<rect x="24" y="31" width="30" height="7" fill="' + lite + '"/><rect x="24" y="31" width="30" height="1.5" fill="' + dark + '" opacity="0.7"/>' : '') + ((m & 8) ? '<rect x="-6" y="31" width="30" height="7" fill="' + lite + '"/><rect x="-6" y="31" width="30" height="1.5" fill="' + dark + '" opacity="0.7"/>' : '') + '</g>';
      if (w) { const horiz = (m & 2) || (m & 8) || !(m & 5); o += horiz ? '<rect x="14" y="13" width="20" height="18" fill="' + INK + '"/><rect x="16" y="15" width="16" height="14" fill="#cfe6f6"/><path d="M24 15 v14 M16 22 h16" stroke="#6b4a2e" stroke-width="2"/><rect x="16" y="15" width="16" height="3" fill="#fff" opacity="0.35"/>' : '<rect x="13" y="14" width="22" height="20" fill="' + INK + '"/><rect x="15" y="16" width="18" height="16" fill="#cfe6f6"/><path d="M24 16 v16 M15 24 h18" stroke="#6b4a2e" stroke-width="2"/><rect x="15" y="16" width="18" height="4" fill="#fff" opacity="0.35"/>'; }
      c += spr('wl' + mat[0] + m + w, 48, 48, o); } }
  // a windbreak: two or three hides draped over a leaning frame of branches, pegged at the foot
  c += spr('tent0', 48, 44, '<ellipse cx="24" cy="40" rx="20" ry="3.5" fill="#000" opacity="0.28"/><path d="M4 38 L10 12 L26 6 L44 10 L42 38 Z" fill="' + INK + '"/><path d="M6 36 L11 14 L26 8 L42 12 L40 36 Z" fill="#8a6a46"/><path d="M11 14 L26 8 L28 36 L6 36 Z" fill="#9d7a52"/><path d="M26 8 L42 12 L40 36 L28 36 Z" fill="#7a5a3c"/><path d="M14 16 q6 10 2 20 M30 12 q4 11 1 23" stroke="#5e4330" stroke-width="1.2" fill="none" opacity="0.7"/><path d="M10 12 L6 40 M26 6 L28 40 M44 10 L42 40" stroke="#4a3522" stroke-width="2" fill="none"/><path d="M4 39 h6 M24 40 h8 M40 39 h6" stroke="' + INK + '" stroke-width="2"/><ellipse cx="18" cy="24" rx="4" ry="2.5" fill="#b08a62" opacity="0.5"/>');
  c += '.fire0::before,.fire1::before,.fire0::after,.fire1::after{content:"";position:absolute;left:0;top:0;width:100%;height:100%;background-size:100% 100%;background-repeat:no-repeat;transform-origin:50% 94%;pointer-events:none}';
  c += '.fire0::before{background-image:' + svgBg(40, 50, FIRE_M(0)) + ';animation:flameB .42s ease-in-out infinite alternate}.fire1::before{background-image:' + svgBg(40, 50, FIRE_M(1)) + ';animation:flameB .5s ease-in-out -.2s infinite alternate}';
  c += '.fire0::after,.fire1::after{background-image:' + svgBg(40, 50, FIRE_C) + ';animation:flameC .3s ease-in-out infinite alternate}';
  c += '.fire0,.fire1{position:relative}';
  return c;
})();
