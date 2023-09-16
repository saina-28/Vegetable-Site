const express = require('express');
const { signup, getUserProfile, updateUserProfile, deleteUser} = require('../controllers/userController');
const auth = require("../middlewares/auth");
const router = express.Router()


router.post('/signup', signup)
router.post('/get-user-profile', auth,  getUserProfile)
router.post('/update-user-profile', auth, updateUserProfile)
router.post('/delete-user', deleteUser)

module.exports = router;