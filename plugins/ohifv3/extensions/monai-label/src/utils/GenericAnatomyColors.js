function componentToHex(c) {
  const hex = c.toString(16);
  return hex.length === 1 ? '0' + hex : hex;
}

function rgbToHex(r, g, b) {
  return '#' + componentToHex(r) + componentToHex(g) + componentToHex(b);
}

export const GenericNames = [
  'james',
  'robert',
  'john',
  'michael',
  'william',
  'david',
  'richard',
  'joseph',
  'thomas',
  'charles',
  'christopher',
  'daniel',
  'matthew',
  'anthony',
  'mark',
  'donald',
  'steven',
  'paul',
  'andrew',
  'joshua',
  'kenneth',
  'kevin',
  'brian',
  'george',
  'edward',
  'ronald',
  'timothy',
  'jason',
  'jeffrey',
  'ryan',
  'jacob',
  'gary',
  'nicholas',
  'eric',
  'jonathan',
  'stephen',
  'larry',
  'justin',
  'scott',
  'brandon',
  'benjamin',
  'samuel',
  'gregory',
  'frank',
  'alexander',
  'raymond',
  'patrick',
  'jack',
  'dennis',
  'jerry',
  'tyler',
  'aaron',
  'jose',
  'adam',
  'henry',
  'nathan',
  'douglas',
  'zachary',
  'peter',
  'kyle',
  'walter',
  'ethan',
  'jeremy',
  'harold',
  'keith',
  'christian',
  'roger',
  'noah',
  'gerald',
  'carl',
  'terry',
  'sean',
  'austin',
  'arthur',
  'lawrence',
  'jesse',
  'dylan',
  'bryan',
  'joe',
  'jordan',
  'billy',
  'bruce',
  'albert',
  'willie',
  'gabriel',
  'logan',
  'alan',
  'juan',
  'wayne',
  'roy',
  'ralph',
  'randy',
  'eugene',
  'vincent',
  'russell',
  'elijah',
  'louis',
  'bobby',
  'philip',
  'johnny',
  'mary',
  'patricia',
  'jennifer',
  'linda',
  'elizabeth',
  'barbara',
  'susan',
  'jessica',
  'sarah',
  'karen',
  'nancy',
  'lisa',
  'betty',
  'margaret',
  'sandra',
  'ashley',
  'kimberly',
  'emily',
  'donna',
  'michelle',
  'dorothy',
  'carol',
  'amanda',
  'melissa',
  'deborah',
  'stephanie',
  'rebecca',
  'sharon',
  'laura',
  'cynthia',
  'kathleen',
  'amy',
  'shirley',
  'angela',
  'helen',
  'anna',
  'brenda',
  'pamela',
  'nicole',
  'emma',
  'samantha',
  'katherine',
  'christine',
  'debra',
  'rachel',
  'catherine',
  'carolyn',
  'janet',
  'ruth',
  'maria',
  'heather',
  'diane',
  'virginia',
  'julie',
  'joyce',
  'victoria',
  'olivia',
  'kelly',
  'christina',
  'lauren',
  'joan',
  'evelyn',
  'judith',
  'megan',
  'cheryl',
  'andrea',
  'hannah',
  'martha',
  'jacqueline',
  'frances',
  'gloria',
  'ann',
  'teresa',
  'kathryn',
  'sara',
  'janice',
  'jean',
  'alice',
  'madison',
  'doris',
  'abigail',
  'julia',
  'judy',
  'grace',
  'denise',
  'amber',
  'marilyn',
  'beverly',
  'danielle',
  'theresa',
  'sophia',
  'marie',
  'diana',
  'brittany',
  'natalie',
  'isabella',
  'charlotte',
  'rose',
  'alexis',
  'kayla',
];

export const GenericAnatomyColors = [
  { label: 'background', value: rgbToHex(0, 0, 0) },
  { label: 'tissue', value: rgbToHex(128, 174, 128) },
  { label: 'bone', value: rgbToHex(241, 214, 145) },
  { label: 'skin', value: rgbToHex(177, 122, 101) },
  { label: 'connective tissue', value: rgbToHex(111, 184, 210) },
  { label: 'blood', value: rgbToHex(216, 101, 79) },
  { label: 'organ', value: rgbToHex(221, 130, 101) },
  { label: 'mass', value: rgbToHex(144, 238, 144) },
  { label: 'muscle', value: rgbToHex(192, 104, 88) },
  { label: 'foreign object', value: rgbToHex(220, 245, 20) },
  { label: 'waste', value: rgbToHex(78, 63, 0) },
  { label: 'teeth', value: rgbToHex(255, 250, 220) },
  { label: 'fat', value: rgbToHex(230, 220, 70) },
  { label: 'gray matter', value: rgbToHex(200, 200, 235) },
  { label: 'white matter', value: rgbToHex(250, 250, 210) },
  { label: 'nerve', value: rgbToHex(244, 214, 49) },
  { label: 'vein', value: rgbToHex(0, 151, 206) },
  { label: 'artery', value: rgbToHex(216, 101, 79) },
  { label: 'capillary', value: rgbToHex(183, 156, 220) },
  { label: 'ligament', value: rgbToHex(183, 214, 211) },
  { label: 'tendon', value: rgbToHex(152, 189, 207) },
  { label: 'cartilage', value: rgbToHex(111, 184, 210) },
  { label: 'meniscus', value: rgbToHex(178, 212, 242) },
  { label: 'lymph node', value: rgbToHex(68, 172, 100) },
  { label: 'lymphatic vessel', value: rgbToHex(111, 197, 131) },
  { label: 'cerebro-spinal fluid', value: rgbToHex(85, 188, 255) },
  { label: 'bile', value: rgbToHex(0, 145, 30) },
  { label: 'urine', value: rgbToHex(214, 230, 130) },
  { label: 'feces', value: rgbToHex(78, 63, 0) },
  { label: 'gas', value: rgbToHex(218, 255, 255) },
  { label: 'fluid', value: rgbToHex(170, 250, 250) },
  { label: 'edema', value: rgbToHex(140, 224, 228) },
  { label: 'bleeding', value: rgbToHex(188, 65, 28) },
  { label: 'necrosis', value: rgbToHex(216, 191, 216) },
  { label: 'clot', value: rgbToHex(145, 60, 66) },
  { label: 'embolism', value: rgbToHex(150, 98, 83) },
  { label: 'head', value: rgbToHex(177, 122, 101) },
  { label: 'central nervous system', value: rgbToHex(244, 214, 49) },
  { label: 'brain', value: rgbToHex(250, 250, 225) },
  { label: 'gray matter of brain', value: rgbToHex(200, 200, 215) },
  { label: 'telencephalon', value: rgbToHex(68, 131, 98) },
  { label: 'cerebral cortex', value: rgbToHex(128, 174, 128) },
  { label: 'right frontal lobe', value: rgbToHex(83, 146, 164) },
  { label: 'left frontal lobe', value: rgbToHex(83, 146, 164) },
  { label: 'right temporal lobe', value: rgbToHex(162, 115, 105) },
  { label: 'left temporal lobe', value: rgbToHex(162, 115, 105) },
  { label: 'right parietal lobe', value: rgbToHex(141, 93, 137) },
  { label: 'left parietal lobe', value: rgbToHex(141, 93, 137) },
  { label: 'right occipital lobe', value: rgbToHex(182, 166, 110) },
  { label: 'left occipital lobe', value: rgbToHex(182, 166, 110) },
  { label: 'right insular lobe', value: rgbToHex(188, 135, 166) },
  { label: 'left insular lobe', value: rgbToHex(188, 135, 166) },
  { label: 'right limbic lobe', value: rgbToHex(154, 150, 201) },
  { label: 'left limbic lobe', value: rgbToHex(154, 150, 201) },
  { label: 'right striatum', value: rgbToHex(177, 140, 190) },
  { label: 'left striatum', value: rgbToHex(177, 140, 190) },
  { label: 'right caudate nucleus', value: rgbToHex(30, 111, 85) },
  { label: 'left caudate nucleus', value: rgbToHex(30, 111, 85) },
  { label: 'right putamen', value: rgbToHex(210, 157, 166) },
  { label: 'left putamen', value: rgbToHex(210, 157, 166) },
  { label: 'right pallidum', value: rgbToHex(48, 129, 126) },
  { label: 'left pallidum', value: rgbToHex(48, 129, 126) },
  { label: 'right amygdaloid complex', value: rgbToHex(98, 153, 112) },
  { label: 'left amygdaloid complex', value: rgbToHex(98, 153, 112) },
  { label: 'diencephalon', value: rgbToHex(69, 110, 53) },
  { label: 'thalamus', value: rgbToHex(166, 113, 137) },
  { label: 'right thalamus', value: rgbToHex(122, 101, 38) },
  { label: 'left thalamus', value: rgbToHex(122, 101, 38) },
  { label: 'pineal gland', value: rgbToHex(253, 135, 192) },
  { label: 'midbrain', value: rgbToHex(145, 92, 109) },
  { label: 'substantia nigra', value: rgbToHex(46, 101, 131) },
  { label: 'right substantia nigra', value: rgbToHex(0, 108, 112) },
  { label: 'left substantia nigra', value: rgbToHex(0, 108, 112) },
  { label: 'cerebral white matter', value: rgbToHex(250, 250, 225) },
  {
    label: 'right superior longitudinal fasciculus',
    value: rgbToHex(127, 150, 88),
  },
  {
    label: 'left superior longitudinal fasciculus',
    value: rgbToHex(127, 150, 88),
  },
  {
    label: 'right inferior longitudinal fasciculus',
    value: rgbToHex(159, 116, 163),
  },
  {
    label: 'left inferior longitudinal fasciculus',
    value: rgbToHex(159, 116, 163),
  },
  { label: 'right arcuate fasciculus', value: rgbToHex(125, 102, 154) },
  { label: 'left arcuate fasciculus', value: rgbToHex(125, 102, 154) },
  { label: 'right uncinate fasciculus', value: rgbToHex(106, 174, 155) },
  { label: 'left uncinate fasciculus', value: rgbToHex(106, 174, 155) },
  { label: 'right cingulum bundle', value: rgbToHex(154, 146, 83) },
  { label: 'left cingulum bundle', value: rgbToHex(154, 146, 83) },
  { label: 'projection fibers', value: rgbToHex(126, 126, 55) },
  { label: 'right corticospinal tract', value: rgbToHex(201, 160, 133) },
  { label: 'left corticospinal tract', value: rgbToHex(201, 160, 133) },
  { label: 'right optic radiation', value: rgbToHex(78, 152, 141) },
  { label: 'left optic radiation', value: rgbToHex(78, 152, 141) },
  { label: 'right medial lemniscus', value: rgbToHex(174, 140, 103) },
  { label: 'left medial lemniscus', value: rgbToHex(174, 140, 103) },
  {
    label: 'right superior cerebellar peduncle',
    value: rgbToHex(139, 126, 177),
  },
  {
    label: 'left superior cerebellar peduncle',
    value: rgbToHex(139, 126, 177),
  },
  { label: 'right middle cerebellar peduncle', value: rgbToHex(148, 120, 72) },
  { label: 'left middle cerebellar peduncle', value: rgbToHex(148, 120, 72) },
  {
    label: 'right inferior cerebellar peduncle',
    value: rgbToHex(186, 135, 135),
  },
  {
    label: 'left inferior cerebellar peduncle',
    value: rgbToHex(186, 135, 135),
  },
  { label: 'optic chiasm', value: rgbToHex(99, 106, 24) },
  { label: 'right optic tract', value: rgbToHex(156, 171, 108) },
  { label: 'left optic tract', value: rgbToHex(156, 171, 108) },
  { label: 'right fornix', value: rgbToHex(64, 123, 147) },
  { label: 'left fornix', value: rgbToHex(64, 123, 147) },
  { label: 'commissural fibers', value: rgbToHex(138, 95, 74) },
  { label: 'corpus callosum', value: rgbToHex(97, 113, 158) },
  { label: 'posterior commissure', value: rgbToHex(126, 161, 197) },
  { label: 'cerebellar white matter', value: rgbToHex(194, 195, 164) },
  { label: 'CSF space', value: rgbToHex(85, 188, 255) },
  { label: 'ventricles of brain', value: rgbToHex(88, 106, 215) },
  { label: 'right lateral ventricle', value: rgbToHex(88, 106, 215) },
  { label: 'left lateral ventricle', value: rgbToHex(88, 106, 215) },
  { label: 'right third ventricle', value: rgbToHex(88, 106, 215) },
  { label: 'left third ventricle', value: rgbToHex(88, 106, 215) },
  { label: 'cerebral aqueduct', value: rgbToHex(88, 106, 215) },
  { label: 'fourth ventricle', value: rgbToHex(88, 106, 215) },
  { label: 'subarachnoid space', value: rgbToHex(88, 106, 215) },
  { label: 'spinal cord', value: rgbToHex(244, 214, 49) },
  { label: 'gray matter of spinal cord', value: rgbToHex(200, 200, 215) },
  { label: 'white matter of spinal cord', value: rgbToHex(250, 250, 225) },
  { label: 'endocrine system of brain', value: rgbToHex(82, 174, 128) },
  { label: 'pituitary gland', value: rgbToHex(57, 157, 110) },
  { label: 'adenohypophysis', value: rgbToHex(60, 143, 83) },
  { label: 'neurohypophysis', value: rgbToHex(92, 162, 109) },
  { label: 'meninges', value: rgbToHex(255, 244, 209) },
  { label: 'dura mater', value: rgbToHex(255, 244, 209) },
  { label: 'arachnoid', value: rgbToHex(255, 244, 209) },
  { label: 'pia mater', value: rgbToHex(255, 244, 209) },
  { label: 'muscles of head', value: rgbToHex(201, 121, 77) },
  { label: 'salivary glands', value: rgbToHex(70, 163, 117) },
  { label: 'lips', value: rgbToHex(188, 91, 95) },
  { label: 'nose', value: rgbToHex(177, 122, 101) },
  { label: 'tongue', value: rgbToHex(166, 84, 94) },
  { label: 'soft palate', value: rgbToHex(182, 105, 107) },
  { label: 'right inner ear', value: rgbToHex(229, 147, 118) },
  { label: 'left inner ear', value: rgbToHex(229, 147, 118) },
  { label: 'right external ear', value: rgbToHex(174, 122, 90) },
  { label: 'left external ear', value: rgbToHex(174, 122, 90) },
  { label: 'right middle ear', value: rgbToHex(201, 112, 73) },
  { label: 'left middle ear', value: rgbToHex(201, 112, 73) },
  { label: 'right eyeball', value: rgbToHex(194, 142, 0) },
  { label: 'left eyeball', value: rgbToHex(194, 142, 0) },
  { label: 'skull', value: rgbToHex(241, 213, 144) },
  { label: 'right frontal bone', value: rgbToHex(203, 179, 77) },
  { label: 'left frontal bone', value: rgbToHex(203, 179, 77) },
  { label: 'right parietal bone', value: rgbToHex(229, 204, 109) },
  { label: 'left parietal bone', value: rgbToHex(229, 204, 109) },
  { label: 'right temporal bone', value: rgbToHex(255, 243, 152) },
  { label: 'left temporal bone', value: rgbToHex(255, 243, 152) },
  { label: 'right sphenoid bone', value: rgbToHex(209, 185, 85) },
  { label: 'left sphenoid bone', value: rgbToHex(209, 185, 85) },
  { label: 'right ethmoid bone', value: rgbToHex(248, 223, 131) },
  { label: 'left ethmoid bone', value: rgbToHex(248, 223, 131) },
  { label: 'occipital bone', value: rgbToHex(255, 230, 138) },
  { label: 'maxilla', value: rgbToHex(196, 172, 68) },
  { label: 'right zygomatic bone', value: rgbToHex(255, 255, 167) },
  { label: 'right lacrimal bone', value: rgbToHex(255, 250, 160) },
  { label: 'vomer bone', value: rgbToHex(255, 237, 145) },
  { label: 'right palatine bone', value: rgbToHex(242, 217, 123) },
  { label: 'left palatine bone', value: rgbToHex(242, 217, 123) },
  { label: 'mandible', value: rgbToHex(222, 198, 101) },
  { label: 'neck', value: rgbToHex(177, 122, 101) },
  { label: 'muscles of neck', value: rgbToHex(213, 124, 109) },
  { label: 'pharynx', value: rgbToHex(184, 105, 108) },
  { label: 'larynx', value: rgbToHex(150, 208, 243) },
  { label: 'thyroid gland', value: rgbToHex(62, 162, 114) },
  { label: 'right parathyroid glands', value: rgbToHex(62, 162, 114) },
  { label: 'left parathyroid glands', value: rgbToHex(62, 162, 114) },
  { label: 'skeleton of neck', value: rgbToHex(242, 206, 142) },
  { label: 'hyoid bone', value: rgbToHex(250, 210, 139) },
  { label: 'cervical vertebral column', value: rgbToHex(255, 255, 207) },
  { label: 'thorax', value: rgbToHex(177, 122, 101) },
  { label: 'trachea', value: rgbToHex(182, 228, 255) },
  { label: 'bronchi', value: rgbToHex(175, 216, 244) },
  { label: 'lung', value: rgbToHex(197, 165, 145) },
  { label: 'lung tumor', value: rgbToHex(144, 238, 144) },
  { label: 'right lung', value: rgbToHex(197, 165, 145) },
  { label: 'left lung', value: rgbToHex(197, 165, 145) },
  { label: 'superior lobe of right lung', value: rgbToHex(172, 138, 115) },
  { label: 'superior lobe of left lung', value: rgbToHex(172, 138, 115) },
  { label: 'middle lobe of right lung', value: rgbToHex(202, 164, 140) },
  { label: 'inferior lobe of right lung', value: rgbToHex(224, 186, 162) },
  { label: 'inferior lobe of left lung', value: rgbToHex(224, 186, 162) },
  { label: 'pleura', value: rgbToHex(255, 245, 217) },
  { label: 'heart', value: rgbToHex(206, 110, 84) },
  { label: 'right atrium', value: rgbToHex(210, 115, 89) },
  { label: 'left atrium', value: rgbToHex(203, 108, 81) },
  { label: 'atrial septum', value: rgbToHex(233, 138, 112) },
  { label: 'ventricular septum', value: rgbToHex(195, 100, 73) },
  { label: 'right ventricle of heart', value: rgbToHex(181, 85, 57) },
  { label: 'left ventricle of heart', value: rgbToHex(152, 55, 13) },
  { label: 'mitral valve', value: rgbToHex(159, 63, 27) },
  { label: 'tricuspid valve', value: rgbToHex(166, 70, 38) },
  { label: 'aortic valve', value: rgbToHex(218, 123, 97) },
  { label: 'pulmonary valve', value: rgbToHex(225, 130, 104) },
  { label: 'aorta', value: rgbToHex(224, 97, 76) },
  { label: 'pericardium', value: rgbToHex(255, 244, 209) },
  { label: 'pericardial cavity', value: rgbToHex(184, 122, 154) },
  { label: 'esophagus', value: rgbToHex(211, 171, 143) },
  { label: 'thymus', value: rgbToHex(47, 150, 103) },
  { label: 'mediastinum', value: rgbToHex(255, 244, 209) },
  { label: 'skin of thoracic wall', value: rgbToHex(173, 121, 88) },
  { label: 'muscles of thoracic wall', value: rgbToHex(188, 95, 76) },
  { label: 'skeleton of thorax', value: rgbToHex(255, 239, 172) },
  { label: 'thoracic vertebral column', value: rgbToHex(226, 202, 134) },
  { label: 'ribs', value: rgbToHex(253, 232, 158) },
  { label: 'sternum', value: rgbToHex(244, 217, 154) },
  { label: 'right clavicle', value: rgbToHex(205, 179, 108) },
  { label: 'left clavicle', value: rgbToHex(205, 179, 108) },
  { label: 'abdominal cavity', value: rgbToHex(186, 124, 161) },
  { label: 'abdomen', value: rgbToHex(177, 122, 101) },
  { label: 'peritoneum', value: rgbToHex(255, 255, 220) },
  { label: 'omentum', value: rgbToHex(234, 234, 194) },
  { label: 'peritoneal cavity', value: rgbToHex(204, 142, 178) },
  { label: 'retroperitoneal space', value: rgbToHex(180, 119, 153) },
  { label: 'stomach', value: rgbToHex(216, 132, 105) },
  { label: 'duodenum', value: rgbToHex(255, 253, 229) },
  { label: 'small bowel', value: rgbToHex(205, 167, 142) },
  { label: 'colon', value: rgbToHex(204, 168, 143) },
  { label: 'anus', value: rgbToHex(255, 224, 199) },
  { label: 'liver', value: rgbToHex(221, 130, 101) },
  { label: 'liver tumor', value: rgbToHex(144, 238, 144) },
  { label: 'biliary tree', value: rgbToHex(0, 145, 30) },
  { label: 'gallbladder', value: rgbToHex(139, 150, 98) },
  { label: 'pancreas', value: rgbToHex(249, 180, 111) },
  { label: 'pancreatic tumor', value: rgbToHex(144, 238, 144) },
  { label: 'spleen', value: rgbToHex(157, 108, 162) },
  { label: 'urinary system', value: rgbToHex(203, 136, 116) },
  { label: 'kidney', value: rgbToHex(185, 102, 83) },
  { label: 'kidney tumor', value: rgbToHex(144, 238, 144) },
  { label: 'right kidney', value: rgbToHex(185, 102, 83) },
  { label: 'left kidney', value: rgbToHex(185, 102, 83) },
  { label: 'right ureter', value: rgbToHex(247, 182, 164) },
  { label: 'left ureter', value: rgbToHex(247, 182, 164) },
  { label: 'urinary bladder', value: rgbToHex(222, 154, 132) },
  { label: 'urethra', value: rgbToHex(124, 186, 223) },
  { label: 'right adrenal gland', value: rgbToHex(249, 186, 150) },
  { label: 'left adrenal gland', value: rgbToHex(249, 186, 150) },
  { label: 'female internal genitalia', value: rgbToHex(244, 170, 147) },
  { label: 'uterus', value: rgbToHex(255, 181, 158) },
  { label: 'right fallopian tube', value: rgbToHex(255, 190, 165) },
  { label: 'left fallopian tube', value: rgbToHex(227, 153, 130) },
  { label: 'right ovary', value: rgbToHex(213, 141, 113) },
  { label: 'left ovary', value: rgbToHex(213, 141, 113) },
  { label: 'vagina', value: rgbToHex(193, 123, 103) },
  { label: 'male internal genitalia', value: rgbToHex(216, 146, 127) },
  { label: 'prostate', value: rgbToHex(230, 158, 140) },
  { label: 'right seminal vesicle', value: rgbToHex(245, 172, 147) },
  { label: 'left seminal vesicle', value: rgbToHex(245, 172, 147) },
  { label: 'right deferent duct', value: rgbToHex(241, 172, 151) },
  { label: 'left deferent duct', value: rgbToHex(241, 172, 151) },
  { label: 'skin of abdominal wall', value: rgbToHex(177, 124, 92) },
  { label: 'muscles of abdominal wall', value: rgbToHex(171, 85, 68) },
  { label: 'skeleton of abdomen', value: rgbToHex(217, 198, 131) },
  { label: 'lumbar vertebral column', value: rgbToHex(212, 188, 102) },
  { label: 'female external genitalia', value: rgbToHex(185, 135, 134) },
  { label: 'male external genitalia', value: rgbToHex(185, 135, 134) },
  { label: 'skeleton of upper limb', value: rgbToHex(198, 175, 125) },
  { label: 'muscles of upper limb', value: rgbToHex(194, 98, 79) },
  { label: 'right upper limb', value: rgbToHex(177, 122, 101) },
  { label: 'left upper limb', value: rgbToHex(177, 122, 101) },
  { label: 'right shoulder', value: rgbToHex(177, 122, 101) },
  { label: 'left shoulder', value: rgbToHex(177, 122, 101) },
  { label: 'right arm', value: rgbToHex(177, 122, 101) },
  { label: 'left arm', value: rgbToHex(177, 122, 101) },
  { label: 'right elbow', value: rgbToHex(177, 122, 101) },
  { label: 'left elbow', value: rgbToHex(177, 122, 101) },
  { label: 'right forearm', value: rgbToHex(177, 122, 101) },
  { label: 'left forearm', value: rgbToHex(177, 122, 101) },
  { label: 'right wrist', value: rgbToHex(177, 122, 101) },
  { label: 'left wrist', value: rgbToHex(177, 122, 101) },
  { label: 'right hand', value: rgbToHex(177, 122, 101) },
  { label: 'left hand', value: rgbToHex(177, 122, 101) },
  { label: 'skeleton of lower limb', value: rgbToHex(255, 238, 170) },
  { label: 'muscles of lower limb', value: rgbToHex(206, 111, 93) },
  { label: 'right lower limb', value: rgbToHex(177, 122, 101) },
  { label: 'left lower limb', value: rgbToHex(177, 122, 101) },
  { label: 'right hip', value: rgbToHex(177, 122, 101) },
  { label: 'left hip', value: rgbToHex(177, 122, 101) },
  { label: 'right thigh', value: rgbToHex(177, 122, 101) },
  { label: 'left thigh', value: rgbToHex(177, 122, 101) },
  { label: 'right knee', value: rgbToHex(177, 122, 101) },
  { label: 'left knee', value: rgbToHex(177, 122, 101) },
  { label: 'right leg', value: rgbToHex(177, 122, 101) },
  { label: 'left leg', value: rgbToHex(177, 122, 101) },
  { label: 'right foot', value: rgbToHex(177, 122, 101) },
  { label: 'left foot', value: rgbToHex(177, 122, 101) },
  { label: 'peripheral nervous system', value: rgbToHex(216, 186, 0) },
  { label: 'autonomic nerve', value: rgbToHex(255, 226, 77) },
  { label: 'sympathetic trunk', value: rgbToHex(255, 243, 106) },
  { label: 'cranial nerves', value: rgbToHex(255, 234, 92) },
  { label: 'vagus nerve', value: rgbToHex(240, 210, 35) },
  { label: 'peripheral nerve', value: rgbToHex(224, 194, 0) },
  { label: 'circulatory system', value: rgbToHex(213, 99, 79) },
  { label: 'systemic arterial system', value: rgbToHex(217, 102, 81) },
  { label: 'systemic venous system', value: rgbToHex(0, 147, 202) },
  { label: 'pulmonary arterial system', value: rgbToHex(0, 122, 171) },
  { label: 'pulmonary venous system', value: rgbToHex(186, 77, 64) },
  { label: 'lymphatic system', value: rgbToHex(111, 197, 131) },
  { label: 'needle', value: rgbToHex(240, 255, 30) },
  { label: 'region 0', value: rgbToHex(185, 232, 61) },
  { label: 'region 1', value: rgbToHex(0, 226, 255) },
  { label: 'region 2', value: rgbToHex(251, 159, 255) },
  { label: 'region 3', value: rgbToHex(230, 169, 29) },
  { label: 'region 4', value: rgbToHex(0, 194, 113) },
  { label: 'region 5', value: rgbToHex(104, 160, 249) },
  { label: 'region 6', value: rgbToHex(221, 108, 158) },
  { label: 'region 7', value: rgbToHex(137, 142, 0) },
  { label: 'region 8', value: rgbToHex(230, 70, 0) },
  { label: 'region 9', value: rgbToHex(0, 147, 0) },
  { label: 'region 10', value: rgbToHex(0, 147, 248) },
  { label: 'region 11', value: rgbToHex(231, 0, 206) },
  { label: 'region 12', value: rgbToHex(129, 78, 0) },
  { label: 'region 13', value: rgbToHex(0, 116, 0) },
  { label: 'region 14', value: rgbToHex(0, 0, 255) },
  { label: 'region 15', value: rgbToHex(157, 0, 0) },
  { label: 'unknown', value: rgbToHex(100, 100, 130) },
  { label: 'cyst', value: rgbToHex(205, 205, 100) },
];
