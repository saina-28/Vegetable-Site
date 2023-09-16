const userModel = require("../models/User")
const bcrypt = require("bcrypt")

const login =  async (req, res) => {
    const { email, password, remember } = req.body
    try{
        // Existing User Check
        const existingUser = await userModel.findOne({ email : email })
        if(!existingUser){
            return res.status(404).json({ message : "User not found" })
        }
        // matching Password
        const matchPassword = await bcrypt.compare(password, existingUser.password);
        if(!matchPassword){
            return res.status(400).json({ message : "Invalid Credentials" })
        }
        // Creating Session
        req.session.userid = existingUser._id;
        res.cookie('userid', existingUser._id, { maxAge: 60*60*1000, httpOnly: true });
        res.status(201).json({ message : "Login Successful "})
    }
    catch(error){
        console.log(error);
        res.status(500).json({ message : "Something went wrong" });
    }
}

const logout = (req, res) => {
    if(req.session.userid || req.cookies.userid){
        res.clearCookie('userid')
        req.session.destroy()
        console.log("Logged Out!")
        res.status(201).json({ message : "Logged out"})
    }
    else{
        res.status(201).json({ message : "Session Not Found!"})
    }
}


module.exports = { login, logout };